"""Step 5 - generate commit messages for the four context variants (RQ3/RQ4).

  diff_only   the diff alone                                      (Table 4 "Diff-Only")
  exemplars   diff + commit-type definitions + top-5 retrieved    (Table 4 "+Exem.")
              historical commits
  issue       diff + commit-type definitions + the issue's        (Table 4 "+Issue")
              title and body
  full        diff + type definitions + exemplars + issue         (Table 4 "Full", our method)

The prompts are in prompts/generate_*.txt and prompts/commit_types.txt. Diffs longer
than 24,000 characters keep their first 70% and last 30%.

Which issue is given to the model is set with --issue_source:
  gold        the issue the PR actually links to
  identified  the top-ranked issue from results/issue_identification/<repo>/<backbone>.json

Usage
-----
  python src/generate_messages.py --variant full --repos pytorch --backbones claude
  python src/generate_messages.py --variant issue --issue_source identified

Output: results/runs/commit_messages/<variant>/<repo>.json (resumable), in the same
format as the published files under results/commit_messages/.
"""
import argparse
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (BACKBONES, DATA, RESULTS, llm, load_json, load_samples, model_id, prompt,  # noqa: E402
                    save_json, truncate_head_tail)

VARIANTS = ("diff_only", "exemplars", "issue", "full")
DIFF_CAP = 24000
COMMIT_TYPES = prompt("commit_types.txt")


def clean_message(text):
    """Strip code fences, preambles ("Here is the commit message:") and markdown emphasis."""
    if not text:
        return ""
    t = text.strip()
    m = re.search(r"```[a-zA-Z]*\s*\n?(.*?)```", t, re.DOTALL)
    if m and m.group(1).strip():
        t = m.group(1).strip()
    lines = t.replace("```", "").strip().split("\n")

    def preamble(s):
        s = s.strip()
        if s in ("---", "***", "___"):
            return True
        return bool(s) and s.endswith(("：", ":")) and any(
            k in s for k in ("提交", "消息", "commit", "message", "以下", "建议", "here"))

    while lines and (preamble(lines[0]) or not lines[0].strip()):
        lines.pop(0)
    while lines and (lines[-1].strip() in ("---", "***", "___") or not lines[-1].strip()):
        lines.pop()
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", "\n".join(lines))
    return re.sub(r"^#{1,6}\s*", "", t, flags=re.MULTILINE).strip()


def format_exemplars(examples):
    if not examples:
        return "(no similar historical commits retrieved)"
    return "\n\n".join(f"--- Example {e['rank']} (similarity={e['similarity']}, "
                       f"source={e['domain']}@{(e.get('commit_id') or '')[:10]}) ---\n{e['message']}"
                       for e in examples)


def format_issues(sample, numbers):
    pool = {c["number"]: c for c in sample["candidate_pool"]}
    parts = [f"Issue #{n}: {(pool[n].get('title') or '').strip()}\n{(pool[n].get('body') or '').strip()}"
             for n in numbers if n in pool]
    return "\n\n".join(parts) if parts else "(no related issue)"


def issue_numbers(sample, source, identified):
    if source == "gold":
        return sample["gold_issue_numbers"]
    ranked = identified.get(sample["pr_number"])
    return ranked[:1] if ranked else []


def build_prompt(variant, diff, exemplars, issue_text):
    if variant == "diff_only":
        return prompt("generate_diff_only.txt").format(diff=diff)
    fields = {"diff": diff, "type_defs": COMMIT_TYPES}
    if variant in ("exemplars", "full"):
        fields["rag_examples"] = format_exemplars(exemplars)
    if variant in ("issue", "full"):
        fields["issue_report"] = issue_text
    return prompt(f"generate_{variant}.txt").format(**fields)


def generate(repo, variant, backbones, issue_source, rag_dir, out_dir, workers, temperature, top_p):
    samples = load_samples(repo)
    rag = {}
    if variant in ("exemplars", "full"):
        rag = {r["pr_number"]: r["examples"] for r in load_json(os.path.join(rag_dir, f"top5_{repo}.json"))}
    identified = {}
    if variant in ("issue", "full") and issue_source == "identified":
        for bb in backbones:
            res = load_json(os.path.join(RESULTS, "issue_identification", repo, f"{bb}.json"))
            identified[bb] = {r["pr_number"]: r["ranked"] for r in res["per_sample"]}
    out_path = os.path.join(out_dir, variant, f"{repo}.json")
    records = {r["pr_number"]: r for r in (load_json(out_path) if os.path.exists(out_path) else [])}
    prompts, tasks = {}, []
    for s in samples:
        diff, truncated = truncate_head_tail(s["diff_text"], DIFF_CAP), len(s["diff_text"]) > DIFF_CAP
        rec = records.setdefault(s["pr_number"], {
            "repo": repo, "pr_number": s["pr_number"], "pr_title": s["pr_title"],
            "commit_datetime": s["commit_datetime"], "gold_issue_numbers": s["gold_issue_numbers"],
            "diff_chars": len(s["diff_text"]), "diff_truncated": truncated, "messages": {}})
        ctx = {}
        if variant in ("exemplars", "full"):
            ctx["exemplars"] = rag.get(s["pr_number"], [])
        for bb in backbones:
            issue_text = None
            if variant in ("issue", "full"):
                issue_text = format_issues(s, issue_numbers(s, issue_source, identified.get(bb, {})))
                if issue_source == "gold":
                    ctx["issue"] = issue_text
                else:
                    ctx.setdefault("issue_by_backbone", {})[bb] = issue_text
            if not rec["messages"].get(bb):
                prompts[(s["pr_number"], bb)] = build_prompt(variant, diff, ctx.get("exemplars"), issue_text)
                tasks.append((s["pr_number"], bb))
        if ctx:
            rec["context"] = ctx
    print(f"[{variant}/{repo}] {len(tasks)} messages to generate")

    def work(task):
        pr, bb = task
        try:
            return pr, bb, clean_message(llm(prompts[task], model_id(bb), temperature=temperature, top_p=top_p)), None
        except Exception as e:  # noqa: BLE001
            return pr, bb, "", str(e)[:200]

    order = [s["pr_number"] for s in samples]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, fut in enumerate(as_completed([ex.submit(work, t) for t in tasks]), 1):
            pr, bb, msg, err = fut.result()
            records[pr]["messages"][bb] = msg
            if err:
                records[pr].setdefault("errors", {})[bb] = err
            if i % 20 == 0 or i == len(tasks):
                save_json([records[p] for p in order], out_path, indent=1)
                print(f"  {i}/{len(tasks)}")
    save_json([records[p] for p in order], out_path, indent=1)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    ap.add_argument("--repos", nargs="+", default=["pytorch", "keras", "deepspeed"])
    ap.add_argument("--backbones", nargs="+", default=BACKBONES)
    ap.add_argument("--issue_source", default="gold", choices=["gold", "identified"])
    ap.add_argument("--rag_dir", default=os.path.join(DATA, "rag"))
    ap.add_argument("--out_dir", default=os.path.join(RESULTS, "runs", "commit_messages"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=None, help="HTTP backend only")
    ap.add_argument("--top_p", type=float, default=None, help="HTTP backend only")
    args = ap.parse_args()
    for repo in args.repos:
        path = generate(repo, args.variant, args.backbones, args.issue_source, args.rag_dir, args.out_dir,
                        args.workers, args.temperature, args.top_p)
        print(f"[{args.variant}/{repo}] -> {path}")


if __name__ == "__main__":
    main()
