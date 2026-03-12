# Evaluation Comparison Report

- **Baseline**: `results/baseline`
- **Candidate**: `results/posttrained`
- **Baseline model**: `mistralai/Mistral-7B-Instruct-v0.3`
- **Candidate model**: `mistralai/Mistral-7B-Instruct-v0.3`
- **Candidate checkpoint**: `checkpoints/post_v1`

## REGRESSIONS DETECTED

- basic_capability
- robustness
- safety

## Accuracy by Task

| Task | Baseline | Candidate | Delta | Status |
|------|----------|-----------|-------|--------|
| basic_capability | 0.9917 | 0.9250 | -0.0667 | REGRESSION |
| robustness | 0.9800 | 0.7900 | -0.1900 | REGRESSION |
| safety | 0.5500 | 0.5000 | -0.0500 | REGRESSION |

## Safety Detail

| Metric | Baseline | Candidate | Delta |
|--------|----------|-----------|-------|
| refuse_correct | 0.0667 | 0.0000 | -0.0667 |
| unsafe_comply | 0.4333 | 0.5000 | +0.0667 **REGRESSION** |
| over_refuse | 0.0167 | 0.0000 | -0.0167 |
| ok_comply | 0.4833 | 0.5000 | +0.0167 |
