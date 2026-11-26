"""Step 2 - build the evaluation sets (Section 4.1, "Candidate Pool Construction").

For every linked PR in raw/<repo>/prs.json:

  1. candidate pool = issues created before the commit and not yet closed at that time;
  2. the gold issue(s) are added if they are not already in the pool;
  3. the pool is capped at 1,000 issues: the gold issues plus the most recently
     created others;
  4. PRs older than the repository's cut-off date, PRs whose gold issue cannot be
     found, and pools with fewer than 20 issues are dropped.

200 PRs are then taken per repository, spread evenly over commit time, and written
to data/<repo>/ in the compact format described in src/common.py.

Usage
-----
  python src/build_dataset.py pytorch                 # build + sample 200 -> data/pytorch
  python src/build_dataset.py pytorch --n 0           # keep every valid PR
"""
import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import REPOS, ROOT, dataset_dir, load_json, save_samples  # noqa: E402

POOL_CAP = 1000
MIN_POOL = 20
POOL_BODY_CAP = 2500
# Commits before these dates are skipped: the issue history before them is too
# sparse to reconstruct which issues were open at commit time.
MIN_COMMIT_TIME = {
    "pytorch": "2020-10-01T00:00:00Z",
    "keras": "2016-01-01T00:00:00Z",
    "deepspeed": "2020-02-01T00:00:00Z",
    "clickhouse": "2018-01-01T00:00:00Z",
    "pulsar": "2018-01-01T00:00:00Z",
    "rocketmq": "2018-01-01T00:00:00Z",
}


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def build_pool(pr, issues_by_number, cap=POOL_CAP):
    """The candidate pool of one PR: issues open at its commit time, gold first."""
    t = ts(pr["commit_datetime"])
    pool = {}
    for it in issues_by_number.values():
        if it["number"] == pr["pr_number"]:
            continue
        if ts(it["created_at"]) > t:
            continue
        if it.get("closed_at") and ts(it["closed_at"]) <= t:
            continue
        pool[it["number"]] = it
    gold = set(pr["gold_issue_numbers"])
    for n in gold:                                    # force-inject the gold issue
        if n not in pool and n in issues_by_number:
            pool[n] = issues_by_number[n]
    gold_in = [c for c in pool.values() if c["number"] in gold]
    others = sorted((c for c in pool.values() if c["number"] not in gold),
                    key=lambda c: c["created_at"], reverse=True)
    return gold_in + others[: max(0, cap - len(gold_in))]


def build(repo_key, cap=POOL_CAP):
    raw = os.path.join(ROOT, "raw", repo_key)
    issues = {i["number"]: i for i in load_json(os.path.join(raw, "issues.json"))}
    prs = load_json(os.path.join(raw, "prs.json"))
    cutoff = ts(MIN_COMMIT_TIME[repo_key])
    out, skipped = [], {"too_old": 0, "no_gold": 0, "pool_small": 0}
    for pr in prs:
        if ts(pr["commit_datetime"]) < cutoff:
            skipped["too_old"] += 1
            continue
        pool = build_pool(pr, issues, cap)
        numbers = {c["number"] for c in pool}
        gold = [n for n in set(pr["gold_issue_numbers"]) if n in numbers]
        if not gold:
            skipped["no_gold"] += 1
            continue
        if len(pool) < MIN_POOL:
            skipped["pool_small"] += 1
            continue
        out.append({
            "pr_number": pr["pr_number"], "pr_title": pr["pr_title"],
            "commit_datetime": pr["commit_datetime"], "diff_text": pr["diff_text"],
            "gold_issue_numbers": gold,
            "candidate_pool": [{"number": c["number"], "title": c["title"],
                                "body": (c.get("body") or "")[:POOL_BODY_CAP],
                                "labels": c.get("labels", [])} for c in pool],
        })
    print(f"[{repo_key}] {len(out)} valid PRs, skipped {skipped}")
    return out, issues


def spread_sample(samples, n):
    """n samples evenly spaced over commit time (deterministic)."""
    if n <= 0 or len(samples) <= n:
        return samples
    ordered = sorted(samples, key=lambda s: s["commit_datetime"])
    step = len(ordered) / n
    picked, seen = [], set()
    for i in range(n):
        s = ordered[int(i * step)]
        if s["pr_number"] not in seen:
            seen.add(s["pr_number"])
            picked.append(s)
    return picked


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo", choices=list(REPOS))
    ap.add_argument("--n", type=int, default=200, help="samples to keep (0 = all)")
    ap.add_argument("--pool_cap", type=int, default=POOL_CAP)
    ap.add_argument("--out", default=None, help="output directory (default data/<repo>)")
    args = ap.parse_args()
    samples, issues = build(args.repo, args.pool_cap)
    picked = spread_sample(samples, args.n)
    n, ni = save_samples(picked, args.out or dataset_dir(args.repo), issues)
    print(f"[{args.repo}] wrote {n} samples, {ni} distinct issues -> {args.out or dataset_dir(args.repo)}")


if __name__ == "__main__":
    main()
