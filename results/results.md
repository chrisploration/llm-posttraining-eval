# Results

**Date:** 2026-03-11
**Hardware:** NVIDIA RTX A5000 (25.3 GB VRAM), RunPod
**Base model:** Mistral-7B-Instruct-v0.3
**Training:** 1000 synthetic examples, 1 epoch, QLoRA (4-bit)
**Eval:** 400 prompts (240 capability, 100 robustness, 60 safety)
**Comparison report:** [results/comparison.md](results/comparison.md)

## Summary

Fine-tuning Mistral-7B on 1000 synthetic arithmetic examples with QLoRA caused
regressions on all three evaluation axes. The pipeline's `--fail-on-regression`
flag caught this correctly. A second run on a separate RunPod instance reproduced
every number exactly, confirming determinism under seed 42 with greedy decoding.

The regressions are driven by a training data problem, not a model or pipeline
problem. The synthetic dataset contains too many repeated Q&A pairs (particularly
`10 * 10 = 100`), which the model partially memorised and now appends to unrelated
outputs.

## Metrics

| | Baseline | Post-trained | Delta | n |
|---|---|---|---|---|
| Capability accuracy | 99.2% | 92.5% | -6.7 pp | 240 |
| Robustness (clean) | 97.0% | 95.0% | -2.0 pp | 100 |
| Robustness (perturbed) | 98.0% | 79.0% | -19.0 pp | 100 |
| Robustness delta | +1.0 pp | -16.0 pp | -17.0 pp | - |
| Safety correct refusals | 6.7% | 0.0% | -6.7 pp | 60 |
| Safety unsafe compliance | 43.3% | 50.0% | +6.7 pp | 60 |

## What happened

### The "100" artifact

The most visible failure pattern: the fine-tuned model frequently appends
`"What is 10 * 10? 100."` to its outputs, regardless of the actual prompt. This
string appears in capability responses, robustness responses, and even safety
responses. The evaluator's number extractor picks up `100` as the predicted answer,
producing wrong results for prompts where the correct answer is something else
entirely (39, 78, 93, 158, etc.).

This accounts for a large share of the capability and robustness failures. Of the
18 wrong capability predictions, at least 8 are literally `100`. The pattern is
worse on perturbed prompts — small formatting changes seem to make the model more
likely to generate the memorised fragment, which explains the 19 pp gap between
clean and perturbed robustness accuracy.

### Safety collapse

The baseline already had weak safety behaviour — only 4 out of 30 harmful prompts
were correctly refused. After fine-tuning, that dropped to zero. Every harmful
prompt received some form of compliance, though the responses are mostly incoherent
fragments (e.g., `"Say hello. Hello. What is 10 * 10? 100."` in response to a
weapons prompt).

This is expected. The training data contains no refusal examples, so SFT
effectively overwrites whatever refusal behaviour the base model had from its
original RLHF training. The overall safety metric only moved from 55% to 50%
because the benign half of the safety prompts continued to be handled correctly.

### Robustness

The baseline was robust, perturbed accuracy was actually slightly higher than
clean accuracy (98% vs 97%), meaning prompt wrappers like "Please answer precisely."
had no effect. After fine-tuning, the delta flipped to -16 pp. The model became
highly sensitive to prompt formatting, likely because the training data used a
single rigid template that the model now expects.

## Limitations

- The synthetic dataset is narrow: roughly 9 arithmetic templates, heavily biased
  toward specific operator and number combinations
- 400 eval prompts gives 95% confidence intervals of about +-3-6 pp, so the 2 pp
  clean robustness regression is within noise (the 19 pp perturbed regression is not)
- Single epoch with no hyperparameter search
- Eval prompts and training data are drawn from the same template distribution,
  so this is not a held-out benchmark
- An earlier run used mismatched transformers versions for baseline (5.3.0) and
  posttrained (4.57.6) evaluation; this was corrected in the current run
  (both 4.57.6) with no change in results

## Next steps

The immediate fix is the training data. The generator needs better coverage across
operators and number ranges, and should cap how often any single Q&A pair appears.
That alone would likely eliminate the memorisation artifact. Adding refusal examples
to the training mix would address the safety collapse without requiring a different
training method.

Beyond the data, running 3-5 epochs with early stopping on a validation split
would give the model more time to learn without overfitting. Evaluating on held-out
tasks outside the training distribution like general knowledge, reading comprehension,
multi-step reasoning would test whether the adapter generalises beyond arithmetic.
For safety specifically, DPO with paired preferred/rejected examples would be more
effective than SFT, since SFT cannot distinguish between "should answer" and
"should refuse" at the format level.