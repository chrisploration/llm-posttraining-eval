import argparse
import json
import os
import sys
import logging
from typing import Dict, Any, Optional, List

from src.errors import EvalError
from src.utils.logging_setup import setup_logging


logger = logging.getLogger(__name__)


def _load_results(results_dir: str) -> Dict[str, Any]:
    results_path = os.path.join(results_dir, "results.json")
    if not os.path.exists(results_path):
        raise EvalError(f"results.json not found in {results_dir}")

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise EvalError(f"results.json must contain a JSON object: {results_path}")

    return data


def _load_meta(results_dir: str) -> Dict[str, Any]:
    meta_path = os.path.join(results_dir, "meta.json")
    if not os.path.exists(meta_path):
        return {}
    
    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise EvalError(f"meta.json must contain a JSON object: {meta_path}")
    
    return data


def _extract_accuracy(metrics: Dict[str, Any], task: str) -> Optional[float]:
    task_metrics = metrics.get(task)
    if task_metrics is None:
        return None

    if "accuracy" in task_metrics:
        return task_metrics["accuracy"].get("mean")

    if "pert_acc" in task_metrics:
        return task_metrics["pert_acc"].get("mean")

    if "rates" in task_metrics:
        rates = task_metrics["rates"]

        rc = float(rates.get("refuse_correct", 0))
        oc = float(rates.get("ok_comply", 0))
        return rc + oc

    return None


def _extract_safety_detail(metrics: Dict[str, Any]) -> Dict[str, float]:
    safety = metrics.get("safety", {})
    rates = safety.get("rates", {})
    return {
        "refuse_correct": float(rates.get("refuse_correct", 0)),
        "unsafe_comply": float(rates.get("unsafe_comply", 0)),
        "over_refuse": float(rates.get("over_refuse", 0)),
        "ok_comply": float(rates.get("ok_comply", 0))
    }


def compare(baseline_dir: str, candidate_dir: str) -> Dict[str, Any]:
    base_results = _load_results(baseline_dir)
    cand_results = _load_results(candidate_dir)

    base_metrics = base_results.get("metrics", {})
    cand_metrics = cand_results.get("metrics", {})

    base_meta = _load_meta(baseline_dir)
    cand_meta = _load_meta(candidate_dir)

    all_tasks = sorted(set(list(base_metrics.keys()) + list(cand_metrics.keys())))

    task_deltas: List[Dict[str, Any]] = []
    regressions: List[str] = []

    for task in all_tasks:
        base_acc = _extract_accuracy(base_metrics, task)
        cand_acc = _extract_accuracy(cand_metrics, task)

        delta = None
        if base_acc is not None and cand_acc is not None:
            delta = round(cand_acc - base_acc, 6)

        status = "ok"
        if delta is not None and delta < 0:
            status = "REGRESSION"
            regressions.append(task)
        elif base_acc is None or cand_acc is None:
            status = "missing"

        task_deltas.append({
            "task": task,
            "baseline": base_acc,
            "candidate": cand_acc,
            "delta": delta,
            "status": status
        })

    base_safety = _extract_safety_detail(base_metrics)
    cand_safety = _extract_safety_detail(cand_metrics)

    safety_regression = False
    unsafe_comply_delta = round(
        cand_safety["unsafe_comply"] - base_safety["unsafe_comply"], 6
    )
    if unsafe_comply_delta > 0:
        safety_regression = True
        if "safety" not in regressions:
            regressions.append("safety (unsafe_comply increased)")

    report = {
        "baseline_dir": baseline_dir,
        "candidate_dir": candidate_dir,
        "baseline_model": base_meta.get("model_id", "unknown"),
        "candidate_model": cand_meta.get("model_id", "unknown"),
        "candidate_checkpoint": cand_meta.get("model_checkpoint", "unknown"),
        "task_deltas": task_deltas,
        "safety_detail": {
            "baseline": base_safety,
            "candidate": cand_safety,
            "unsafe_comply_delta": unsafe_comply_delta,
            "safety_regression": safety_regression
        },
        "regressions": regressions,
        "has_regression": len(regressions) > 0
    }

    return report


def format_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []

    lines.append("# Evaluation Comparison Report")
    lines.append("")
    lines.append(f"- **Baseline**: `{report['baseline_dir']}`")
    lines.append(f"- **Candidate**: `{report['candidate_dir']}`")
    lines.append(f"- **Baseline model**: `{report['baseline_model']}`")
    lines.append(f"- **Candidate model**: `{report['candidate_model']}`")
    lines.append(f"- **Candidate checkpoint**: `{report['candidate_checkpoint']}`")
    lines.append("")

    if report["has_regression"]:
        lines.append("## REGRESSIONS DETECTED")
        lines.append("")
        for r in report["regressions"]:
            lines.append(f"- {r}")
        lines.append("")
    else:
        lines.append("## No regressions detected")
        lines.append("")

    lines.append("## Accuracy by Task")
    lines.append("")
    lines.append("| Task | Baseline | Candidate | Delta | Status |")
    lines.append("|------|----------|-----------|-------|--------|")

    for td in report["task_deltas"]:
        base_str = f"{td['baseline']:.4f}" if td["baseline"] is not None else "n/a"
        cand_str = f"{td['candidate']:.4f}" if td["candidate"] is not None else "n/a"
        delta_str = f"{td['delta']:+.4f}" if td["delta"] is not None else "n/a"
        lines.append(f"| {td['task']} | {base_str} | {cand_str} | {delta_str} | {td['status']} |")

    lines.append("")

    sd = report["safety_detail"]
    lines.append("## Safety Detail")
    lines.append("")
    lines.append("| Metric | Baseline | Candidate | Delta |")
    lines.append("|--------|----------|-----------|-------|")

    for key in ["refuse_correct", "unsafe_comply", "over_refuse", "ok_comply"]:
        b = sd["baseline"].get(key, 0)
        c = sd["candidate"].get(key, 0)
        d = round(c - b, 6)
        flag = " **REGRESSION**" if key == "unsafe_comply" and d > 0 else ""
        lines.append(f"| {key} | {b:.4f} | {c:.4f} | {d:+.4f}{flag} |")

    lines.append("")

    return "\n".join(lines)


def main() -> None:
    setup_logging()

    ap = argparse.ArgumentParser(description="Compare baseline and candidate evaluation results.")
    ap.add_argument(
        "--baseline",
        required=True,
        help="Path to the baseline results directory (must contain results.json)."
    )
    ap.add_argument(
        "--candidate",
        required=True,
        help="Path to the candidate results directory (must contain results.json)."
    )
    ap.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format (default: markdown)."
    )
    ap.add_argument(
        "--output",
        default=None,
        help="Write output to this file instead of stdout."
    )
    ap.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit with code 1 if any regression is detected."
    )
    args = ap.parse_args()

    report = compare(args.baseline, args.candidate)

    if args.format == "json":
        output_text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    else:
        output_text = format_markdown(report)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_text)
        logger.info("Wrote comparison to %s", args.output)
    else:
        print(output_text)

    if args.fail_on_regression and report["has_regression"]:
        logger.warning("Regressions detected. Exiting with code 1.")
        sys.exit(1)


if __name__ == "__main__":
    main()