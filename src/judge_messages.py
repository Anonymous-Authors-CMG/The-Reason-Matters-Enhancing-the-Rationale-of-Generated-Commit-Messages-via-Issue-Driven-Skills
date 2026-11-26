"""Step 6 - score commit messages on What and Why, 0-4 (Section 5.1.3, CMEval rubric).

The judge prompt (prompts/judge_what_why.txt) is CMEval's unified few-shot
chain-of-thought rubric. Each message is scored against its diff (diffs longer than
24,000 characters keep their first 70% and last 30%). Scores are written into the
message file under "scores"[<judge label>] = {"what": {...}, "why": {...}}.

Usage
-----
  python src/judge_messages.py results/runs/commit_messages/full/pytorch.json \\
      --judge_model deepseek-v4-pro --label deepseek-v4-pro
  python src/judge_messages.py results/runs/commit_messages/full/*.json \\
      --judge_model gpt-5.4 --label GPT-5.4
"""
import argparse
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import llm, load_json, load_samples, prompt, save_json, truncate_head_tail  # noqa: E402

DIFF_CAP = 24000
TEMPLATE = prompt("judge_what_why.txt")


def render(diff, message):
    """Fill {diff} and {message}; the template's literal braces are written {{ }}."""
    t = TEMPLATE.replace("{diff}", "\x00D\x00").replace("{message}", "\x00M\x00")
    t = t.replace("{{", "{").replace("}}", "}")
    return t.replace("\x00D\x00", diff).replace("\x00M\x00", message)


def parse_scores(raw):
    """(what, why) as integers 0-4, or (None, None)."""
    import json
    for block in re.findall(r"\{[^{}]*\"What\"[^{}]*\}", raw or "", re.DOTALL):
        try:
            d = json.loads(block)
            what = int(round(float(d.get("What"))))
            why = int(round(float(d.get("Why")))) if d.get("Why") is not None else None
            return what, why
        except Exception:  # noqa: BLE001
            continue
    w = re.findall(r"\"What\"\s*:\s*\"?([0-4])", raw or "")
    y = re.findall(r"\"Why\"\s*:\s*\"?([0-4])", raw or "")
    return (int(w[-1]), int(y[-1])) if w and y else (None, None)


def judge_file(path, judge_model, label, workers):
    records = load_json(path)
    diffs = {s["pr_number"]: s["diff_text"] for s in load_samples(records[0]["repo"])}
    tasks = []
    for r in records:
        sc = r.setdefault("scores", {}).setdefault(label, {"what": {}, "why": {}})
        for bb, msg in r["messages"].items():
            if msg and (sc["what"].get(bb) is None or sc["why"].get(bb) is None):
                tasks.append((r["pr_number"], bb))
    by_pr = {r["pr_number"]: r for r in records}
    print(f"[{os.path.basename(path)}] {len(tasks)} messages to score with {judge_model}")

    def work(task):
        pr, bb = task
        try:
            raw = llm(render(truncate_head_tail(diffs[pr], DIFF_CAP), by_pr[pr]["messages"][bb]), judge_model)
            return pr, bb, *parse_scores(raw)
        except Exception as e:  # noqa: BLE001
            print(f"  PR#{pr} {bb}: {e}")
            return pr, bb, None, None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, fut in enumerate(as_completed([ex.submit(work, t) for t in tasks]), 1):
            pr, bb, what, why = fut.result()
            sc = by_pr[pr]["scores"][label]
            sc["what"][bb], sc["why"][bb] = what, why
            if i % 25 == 0 or i == len(tasks):
                save_json(records, path, indent=1)
    save_json(records, path, indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--judge_model", required=True, help="model id sent to the backend")
    ap.add_argument("--label", default=None, help="key under 'scores' (default: judge_model)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    for f in args.files:
        judge_file(f, args.judge_model, args.label or args.judge_model, args.workers)


if __name__ == "__main__":
    main()
