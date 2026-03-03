import os
import json
import random
import argparse

from typing import List, Dict, Any

DEFAULT_OUT_PATH = os.path.join("data", "synthetic", "train.jsonl")


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _write_jsonl(path: str, rows: List[Dict[str, Any]], *, force: bool) -> None:
    if os.path.exists(path) and not force:
        raise FileExistsError(
            f"Refusing to overwrite existing file: {path}. "
            "Pass --force to overwrite."
        )
    _ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _mk_example(user: str, assistant: str) -> Dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant}
        ]
    }

# each line: {"messages":[{"role":"user","content":...},{"role":"assistant","content":...}]}
def generate_examples(n: int, rng: random.Random) -> List[Dict[str, Any]]:
    templates = [
        lambda: _mk_example("Say hello.", "Hello."),
        lambda: _mk_example("What is the capital of France?", "Paris."),
        lambda: _mk_example("Write one sentence about cats.", "Cats are small domesticated mammals."),
        lambda: _mk_example("Give a short tip for studying.", "Use spaced repetition."),
        lambda: _mk_example("Define photosynthesis.", "Photosynthesis converts light energy into chemical energy in plants."),
        lambda: _mk_example("Name a primary color.", "Red."),
        lambda: _mk_example(f"What is {rng.randint(2, 99)} + {rng.randint(2, 99)}?", None),
        lambda: _mk_example(f"What is {rng.randint(2, 99)} - {rng.randint(2, 99)}?", None),
        lambda: _mk_example(f"What is {rng.randint(2, 20)} * {rng.randint(2, 20)}?", None)
    ]

    rows: List[Dict[str, Any]] = []
    for _ in range(n):
        ex = rng.choice(templates)()

        # Fill arithmetic answers if needed
        user_msg = ex["messages"][0]["content"]
        if ex["messages"][1]["content"] is None:
            parts = user_msg.replace("What is ", "").replace("?", "").strip().split()
            a = int(parts[0])
            op = parts[1]
            b = int(parts[2])
            if op == "+":
                ans = str(a + b)
            elif op == "-":
                ans = str(a - b)
            elif op == "*":
                ans = str(a * b)
            else:
                ans = "0"
            ex["messages"][1]["content"] = ans

        rows.append(ex)

    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic chat JSONL for post-training.")
    ap.add_argument("--num_examples", type=int, default=200, help="Number of JSONL lines to generate.")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for deterministic generation.")
    ap.add_argument(
        "--out",
        default=DEFAULT_OUT_PATH,
        help=f"Output JSONL path. Default: {DEFAULT_OUT_PATH}",
    )
    ap.add_argument("--force", action="store_true", help="Overwrite output file if it exists.")
    args = ap.parse_args()

    rng = random.Random(int(args.seed))
    rows = generate_examples(int(args.num_examples), rng)
    _write_jsonl(args.out, rows, force=bool(args.force))
    print(f"Wrote {len(rows)} examples to {args.out}")


if __name__ == "__main__":
    main()