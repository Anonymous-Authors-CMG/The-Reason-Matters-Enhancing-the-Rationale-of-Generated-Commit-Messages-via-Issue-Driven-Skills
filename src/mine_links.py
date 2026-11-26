"""Step 1 - mine merged PRs that are linked to an issue (Section 4.1, "Link Mining").

Two link signals are used:

  1. Text references: the PR body/title says "Fixes #N", "Closes #N" or "Resolves #N"
     (any tense, case-insensitive), the keywords that make GitHub close the issue.
  2. GitHub Issue Events: an issue's "referenced" / "closed" events that carry a
     commit_id, resolved to the merged PR containing that commit.

Everything is written to raw/<repo>/ (git-ignored, several GB for large repos):

  raw/<repo>/issues.json   every issue: number, title, body, labels, created_at, closed_at
  raw/<repo>/prs.json      linked merged PRs: diff, commit time, gold issue numbers

Usage
-----
  export GITHUB_TOKENS=ghp_aaa,ghp_bbb
  python src/mine_links.py issues pytorch           # all issues (list + search backfill)
  python src/mine_links.py prs pytorch --target 600 # "Fixes #N" PRs, with diffs
  python src/mine_links.py events deepspeed --target 200   # extra links from Issue Events
"""
import argparse
import calendar
import datetime as dt
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import github_api as gh  # noqa: E402
from common import REPOS, ROOT, load_json, save_json  # noqa: E402

# Repositories mined for the paper, plus the year their history starts.
START_YEAR = {"pytorch": 2017, "keras": 2015, "deepspeed": 2020,
              "clickhouse": 2016, "pulsar": 2016, "rocketmq": 2018}

FIX_RE = re.compile(r"\b(?:fix(?:es|ed)?|close(?:s|d)?|resolve(?:s|d)?)\s*[:\s]?\s*#(\d+)", re.IGNORECASE)
KEYWORDS = ['"Fixes #"', '"Closes #"', '"Resolves #"', '"Fix #"']
BODY_CAP = 5000


def raw_dir(repo):
    d = os.path.join(ROOT, "raw", repo)
    os.makedirs(d, exist_ok=True)
    return d


def fix_refs(text):
    return sorted({int(x) for x in FIX_RE.findall(text or "")})


def issue_record(it):
    return {"number": it["number"], "title": it["title"], "body": (it.get("body") or "")[:BODY_CAP],
            "created_at": it["created_at"], "closed_at": it.get("closed_at"),
            "labels": [lab["name"] for lab in it.get("labels", [])]}


def months(start_year):
    today = dt.date.today()
    for y in range(start_year, today.year + 1):
        for m in range(1, 13):
            if (y, m) > (today.year, today.month):
                return
            yield f"{y}-{m:02d}-01", f"{y}-{m:02d}-{calendar.monthrange(y, m)[1]:02d}"


def search(query, sort="updated", order="desc"):
    """All results of one Search API query (the API stops at 1,000)."""
    for page in range(1, 11):
        try:
            items = gh.get("/search/issues", {"q": query, "per_page": 100, "page": page,
                                              "sort": sort, "order": order}).get("items", [])
        except Exception as e:  # noqa: BLE001
            print(f"  search failed ({query} p{page}): {e}")
            return
        yield from items
        if len(items) < 100:
            return


# ------------------------------------------------------------------ issues

def mine_issues(repo_key):
    """Every issue of the repository, by listing and then by monthly search.

    The list endpoint only reaches the newest ~1,000 issues, and one search query
    returns at most 1,000 results, so older issues are fetched month by month.
    """
    repo = REPOS[repo_key]
    path = os.path.join(raw_dir(repo_key), "issues.json")
    issues = {i["number"]: i for i in (load_json(path) if os.path.exists(path) else [])}
    for page in range(1, 11):
        batch = gh.get(f"/repos/{repo}/issues", {"state": "all", "sort": "created", "direction": "desc",
                                                 "per_page": 100, "page": page})
        if not batch:
            break
        for it in batch:
            if "pull_request" not in it:
                issues.setdefault(it["number"], issue_record(it))
    print(f"[{repo}] {len(issues)} issues after listing")
    for start, end in months(START_YEAR[repo_key]):
        before = len(issues)
        for it in search(f"repo:{repo} is:issue created:{start}..{end}", sort="created", order="asc"):
            if "pull_request" not in it:
                issues.setdefault(it["number"], issue_record(it))
        if len(issues) > before:
            print(f"  {start[:7]}: +{len(issues) - before} (total {len(issues)})")
            save_json(sorted(issues.values(), key=lambda i: i["number"]), path)
    save_json(sorted(issues.values(), key=lambda i: i["number"]), path)
    print(f"[{repo}] done: {len(issues)} issues -> {path}")


def add_missing_gold(repo_key, prs):
    """Fetch gold issues that the issue listing missed."""
    repo = REPOS[repo_key]
    path = os.path.join(raw_dir(repo_key), "issues.json")
    issues = {i["number"]: i for i in (load_json(path) if os.path.exists(path) else [])}
    missing = sorted({n for p in prs for n in p["gold_issue_numbers"]} - set(issues))
    for n in missing:
        try:
            it = gh.get(f"/repos/{repo}/issues/{n}")
        except Exception as e:  # noqa: BLE001
            print(f"  gold #{n} failed: {e}")
            continue
        if "pull_request" not in it:   # references to other PRs are not issues
            issues[n] = issue_record(it)
    if missing:
        save_json(sorted(issues.values(), key=lambda i: i["number"]), path)
        print(f"[{repo}] fetched {len(missing)} gold issues missing from the listing")


# ------------------------------------------------------------------ PRs (signal 1)

def enrich(repo, pr_number, sha=None):
    """Commit time and diff of a merged PR, or None if it was not merged."""
    pr = gh.get(f"/repos/{repo}/pulls/{pr_number}")
    if not pr.get("merged_at"):
        return None
    sha = sha or pr["merge_commit_sha"] or pr["head"]["sha"]
    commit = gh.get(f"/repos/{repo}/commits/{sha}")
    return {"pr_number": pr_number, "pr_title": pr["title"], "pr_body": (pr.get("body") or "")[:2000],
            "commit_sha": sha, "commit_datetime": commit["commit"]["committer"]["date"],
            "merged_at": pr["merged_at"], "base_sha": pr["base"]["sha"], "head_sha": pr["head"]["sha"],
            "diff_text": gh.get_diff(repo, pr_number)}


def mine_prs(repo_key, target):
    """Merged PRs whose body or title says Fixes/Closes/Resolves #N."""
    repo = REPOS[repo_key]
    path = os.path.join(raw_dir(repo_key), "prs.json")
    prs = load_json(path) if os.path.exists(path) else []
    have = {p["pr_number"] for p in prs}
    found = []

    def collect(items):
        for it in items:
            if it["number"] in have:
                continue
            refs = sorted(set(fix_refs(it.get("body")) + fix_refs(it.get("title"))))
            if refs:
                have.add(it["number"])
                found.append((it["number"], refs))
            if len(found) >= target:
                return True
        return False

    done = any(collect(search(f'repo:{repo} is:pr is:merged {kw} in:body')) for kw in KEYWORDS)
    if not done:   # each query stops at 1,000 hits: continue month by month
        for start, end in months(START_YEAR[repo_key]):
            if any(collect(search(f"repo:{repo} is:pr is:merged {kw} in:body merged:{start}..{end}"))
                   for kw in KEYWORDS):
                break
    print(f"[{repo}] {len(found)} new linked PRs, fetching diffs ...")
    for i, (num, refs) in enumerate(found):
        try:
            rec = enrich(repo, num)
        except Exception as e:  # noqa: BLE001
            print(f"  skip PR#{num}: {e}")
            continue
        if rec:
            rec.update(gold_issue_numbers=refs, source="fixes_keyword")
            prs.append(rec)
        if i % 20 == 0:
            save_json(prs, path)
    save_json(prs, path)
    add_missing_gold(repo_key, prs)
    print(f"[{repo}] done: {len(prs)} linked PRs -> {path}")


# ------------------------------------------------------------------ Issue Events (signal 2)

def mine_events(repo_key, target, workers=10):
    """Closed issues whose events reference a commit, resolved to merged PRs."""
    repo = REPOS[repo_key]
    rd = raw_dir(repo_key)
    prs = load_json(os.path.join(rd, "prs.json"))
    have = {p["pr_number"] for p in prs}
    closed = sorted((i for i in load_json(os.path.join(rd, "issues.json")) if i.get("closed_at")),
                    key=lambda i: i["number"], reverse=True)

    def scan(issue_number):
        out = []
        try:
            events = gh.get(f"/repos/{repo}/issues/{issue_number}/events")
        except Exception:  # noqa: BLE001
            return out
        shas = []
        for e in events:
            if e.get("event") in ("referenced", "closed") and e.get("commit_id") and e["commit_id"] not in shas:
                shas.append(e["commit_id"])
        for sha in shas:
            try:
                for pr in gh.get(f"/repos/{repo}/commits/{sha}/pulls"):
                    if pr.get("merged_at"):
                        out.append((pr["number"], sha, issue_number))
            except Exception:  # noqa: BLE001
                continue
        return out

    added = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for start in range(0, len(closed), 400):
            if added >= target:
                break
            new = {}
            for hits in ex.map(scan, [i["number"] for i in closed[start:start + 400]]):
                for pr_number, sha, issue_number in hits:
                    if pr_number not in have and pr_number not in new:
                        new[pr_number] = (sha, issue_number)

            def fetch(item):
                pr_number, (sha, issue_number) = item
                try:
                    rec = enrich(repo, pr_number, sha)
                except Exception:  # noqa: BLE001
                    return None
                if rec:
                    rec.update(pr_body="", gold_issue_numbers=[issue_number], source="issue_events")
                return rec

            for rec in ex.map(fetch, new.items()):
                if rec and rec["pr_number"] not in have:
                    have.add(rec["pr_number"])
                    prs.append(rec)
                    added += 1
            save_json(prs, os.path.join(rd, "prs.json"))
            print(f"  scanned {min(start + 400, len(closed))}/{len(closed)} closed issues, +{added} PRs")
    print(f"[{repo}] done: {len(prs)} linked PRs")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["issues", "prs", "events"])
    ap.add_argument("repo", choices=list(REPOS))
    ap.add_argument("--target", type=int, default=600, help="new PRs to collect")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    if args.what == "issues":
        mine_issues(args.repo)
    elif args.what == "prs":
        mine_prs(args.repo, args.target)
    else:
        mine_events(args.repo, args.target, args.workers)


if __name__ == "__main__":
    main()
