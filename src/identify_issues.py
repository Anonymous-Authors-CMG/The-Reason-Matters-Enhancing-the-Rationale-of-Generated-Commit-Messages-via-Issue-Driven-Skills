"""Step 3 - identify the issue a diff addresses (Section 4.2, RQ2; Tables 1-2, Fig. 2).

Methods
-------
  progressive      Stage 1: the diff plus issue TITLES only, in chunks; each chunk keeps
                   its best ceil(1.5*N) and a merge pass keeps the top N.
                   Stage 2: the diff plus title AND body of those N, ranked by the LLM.
                   The rest of the pool is appended in BM25 order.         [the paper's method]
  bm25_then_llm    BM25 picks the top N, the LLM re-ranks them with full text (Table 2).
  single_title     BM25 keeps K candidates, the LLM ranks their titles in one call.
  single_full      BM25 keeps K candidates, the LLM ranks them with full text in one call.
  bm25_title       BM25 over titles (no LLM).
  bm25_title_body  BM25 over title + body (no LLM).
  random           random order (no LLM).

Usage
-----
  # Table 1: progressive N=10 on one repository with one backbone
  python src/identify_issues.py --dataset pytorch --backbone claude
  # Table 2 / Fig. 2 on the PyTorch design-study subset
  python src/identify_issues.py --dataset design_study_pytorch --method single_title --k 100 --backbone claude
  python src/identify_issues.py --dataset design_study_pytorch --method bm25_title

Results go to results/runs/ unless --out is given, so the published numbers are
never overwritten. Raw prompts and replies go to logs/.
"""
import argparse
import math
import os
import random
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (LOGS, RESULTS, extract_issue_numbers, extract_json, llm, load_samples,  # noqa: E402
                    model_id, prompt, save_json, truncate_head_tail)

# Prompt budget (characters). Pools already cap bodies at 2,500 characters.
MAX_DIFF_CHARS = 8000
MAX_TITLE_CHARS = 200
MAX_BODY_CHARS = 1500
CHUNK_SIZE = 120          # titles per Stage-1 call
TOP_N = 10                # survivors passed to Stage 2

PROMPT_STAGE1 = prompt("stage1_title_ranking.txt")
PROMPT_STAGE2 = prompt("stage2_full_ranking.txt")
JSON_REMINDER = prompt("json_retry_reminder.txt")


# ------------------------------------------------------------------ metrics

def reciprocal_rank(ranked, gold):
    for i, n in enumerate(ranked):
        if n in gold:
            return 1.0 / (i + 1)
    return 0.0


def recall_at(ranked, gold, k):
    return len(set(ranked[:k]) & gold) / max(1, len(gold))


def metrics(per_sample):
    """MRR, Hit@1 and Recall@{1,5,10,20}, averaged over samples (rounded to 4 places)."""
    n = len(per_sample)
    if not n:
        return {"n": 0}
    gold = [set(r["gold"]) for r in per_sample]
    ranked = [r["ranked"] for r in per_sample]
    out = {"n": n,
           "MRR": round(sum(reciprocal_rank(r, g) for r, g in zip(ranked, gold)) / n, 4),
           "Hit@1": round(sum(1 for r, g in zip(ranked, gold) if r and r[0] in g) / n, 4)}
    for k in (1, 5, 10, 20):
        out[f"Recall@{k}"] = round(sum(recall_at(r, g, k) for r, g in zip(ranked, gold)) / n, 4)
    return out


# ------------------------------------------------------------------ non-LLM rankers

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")
STOPWORDS = set("a an the and or not of in to for with on by as is are was were be been being this that these "
                "those from at if then else do does did done have has had you your i we they them our his her its".split())


def tokenize(text):
    return [t for t in TOKEN_RE.findall((text or "").lower()) if len(t) > 2 and t not in STOPWORDS]


def bm25_rank(sample, field="title_body", k1=1.5, b=0.75):
    """Candidate numbers ranked by BM25 against the diff (first 20,000 characters)."""
    docs = [(c["number"], tokenize(c["title"] if field == "title" else c["title"] + " " + (c.get("body") or "")))
            for c in sample["candidate_pool"]]
    n_docs = len(docs)
    avgdl = sum(len(t) for _, t in docs) / max(1, n_docs)
    df = Counter(term for _, toks in docs for term in set(toks))
    idf = {t: math.log((n_docs - f + 0.5) / (f + 0.5) + 1) for t, f in df.items()}
    query = tokenize(sample["diff_text"][:20000])
    scored = []
    for num, toks in docs:
        tf, dl, s = Counter(toks), len(toks), 0.0
        for t in query:
            if t in tf:
                s += idf.get(t, 0) * (tf[t] * (k1 + 1)) / (tf[t] + k1 * (1 - b + b * dl / avgdl))
        scored.append((num, s))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [num for num, _ in scored]


def random_rank(dataset, seed=42):
    rng = random.Random(seed)
    out = []
    for s in dataset:
        nums = [c["number"] for c in s["candidate_pool"]]
        rng.shuffle(nums)
        out.append({"pr_number": s["pr_number"], "ranked": nums, "gold": s["gold_issue_numbers"]})
    return out


# ------------------------------------------------------------------ LLM stages

def format_titles(cands):
    return "\n".join(f"#{c['number']} | {c['title'].replace(chr(10), ' ').strip()[:MAX_TITLE_CHARS]}" for c in cands)


def format_full(cands):
    return "\n---\n".join(
        f"ISSUE #{c['number']}\nTITLE: {c['title'].replace(chr(10), ' ').strip()[:MAX_TITLE_CHARS]}\n"
        f"BODY: {(c.get('body') or '').strip()[:MAX_BODY_CHARS]}\n" for c in cands)


def ask_for_numbers(text, model, allowed, log_path=None, retries=2, timeout=600):
    """Call the LLM until it returns at least one allowed issue number (or give up)."""
    raw_log, cur = "", text
    for attempt in range(retries + 1):
        out = llm(cur, model, timeout=timeout)
        raw_log += out if attempt == 0 else f"\n\n--- RETRY {attempt} ---\n{out}"
        nums = [n for n in extract_issue_numbers(out) if n in allowed]
        if nums:
            break
        cur = text + JSON_REMINDER
    else:
        nums = []
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("PROMPT:\n" + text + "\n\n==OUTPUT==\n" + raw_log)
    return nums


def ask_once(text, model, allowed, log_path=None):
    """One call, plain JSON parsing, no retry: how the single-stage baselines were run."""
    out = llm(text, model)
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("PROMPT:\n" + text + "\n\n==OUTPUT==\n" + out)
    parsed = extract_json(out)
    parsed = parsed if isinstance(parsed, list) else []
    return [int(str(x).lstrip("#")) for x in parsed if str(x).lstrip("#").isdigit() and int(str(x).lstrip("#")) in allowed]


def stage1(diff, pool, top_n, chunk_size, model, log_prefix=None):
    """Title-only ranking in chunks, then one merge pass down to top_n."""
    chunks = [pool[i:i + chunk_size] for i in range(0, len(pool), chunk_size)] or [pool]
    keep = max(top_n, math.ceil(top_n * 1.5))
    picked = []
    for i, chunk in enumerate(chunks):
        text = PROMPT_STAGE1.format(top_n=keep, diff=truncate_head_tail(diff, MAX_DIFF_CHARS),
                                    candidates=format_titles(chunk))
        picked += ask_for_numbers(text, model, {c["number"] for c in chunk},
                                  log_prefix and f"{log_prefix}_coarse_chunk{i}.txt")[:keep]
    merged = list(dict.fromkeys(picked))
    if len(merged) > top_n:
        by_num = {c["number"]: c for c in pool}
        text = PROMPT_STAGE1.format(top_n=top_n, diff=truncate_head_tail(diff, MAX_DIFF_CHARS),
                                    candidates=format_titles([by_num[n] for n in merged if n in by_num]))
        ranked = ask_for_numbers(text, model, set(merged), log_prefix and f"{log_prefix}_coarse_merge.txt")
        merged = ranked + [n for n in merged if n not in ranked]
    return merged[:top_n]


def stage2(diff, cands, model, log_prefix=None):
    """Full-text ranking of the survivors; anything the model omits keeps its order."""
    text = PROMPT_STAGE2.format(diff=truncate_head_tail(diff, MAX_DIFF_CHARS), candidates=format_full(cands))
    given = [c["number"] for c in cands]
    ranked = ask_for_numbers(text, model, set(given), log_prefix and f"{log_prefix}_fine.txt")
    return ranked + [n for n in given if n not in ranked]


def rank_one(sample, method, model, top_n, chunk_size, k, log_dir):
    pool = sample["candidate_pool"]
    by_num = {c["number"]: c for c in pool}
    bm25 = bm25_rank(sample, "title_body")
    log = log_dir and os.path.join(log_dir, f"pr{sample['pr_number']}")
    stage1_top = None
    if method == "progressive":
        stage1_top = stage1(sample["diff_text"], pool, top_n, chunk_size, model, log)
        ranked = stage2(sample["diff_text"], [by_num[n] for n in stage1_top if n in by_num], model, log)
    elif method == "bm25_then_llm":
        top = [by_num[n] for n in bm25[:top_n]]
        text = PROMPT_STAGE2.format(diff=truncate_head_tail(sample["diff_text"], MAX_DIFF_CHARS),
                                    candidates=format_full(top))
        ranked = ask_once(text, model, {c["number"] for c in top}, log and f"{log}.txt")
        ranked += [c["number"] for c in top if c["number"] not in ranked]
    elif method in ("single_title", "single_full"):
        top = [by_num[n] for n in bm25[:k]]
        if method == "single_title":
            text = PROMPT_STAGE1.format(top_n=k, diff=truncate_head_tail(sample["diff_text"], MAX_DIFF_CHARS),
                                        candidates=format_titles(top))
        else:
            text = PROMPT_STAGE2.format(diff=truncate_head_tail(sample["diff_text"], MAX_DIFF_CHARS),
                                        candidates=format_full(top))
        ranked = ask_once(text, model, {c["number"] for c in top}, log and f"{log}.txt")
    else:
        raise ValueError(method)
    seen = set(ranked)
    ranked += [n for n in bm25 if n not in seen]          # complete the ranking in BM25 order
    rec = {"pr_number": sample["pr_number"], "gold": sample["gold_issue_numbers"], "ranked": ranked}
    if stage1_top is not None:
        rec["stage1_topN"] = stage1_top
    return rec


def run(dataset, method, backbone=None, top_n=TOP_N, chunk_size=CHUNK_SIZE, k=100, workers=4, out=None, tag=""):
    if method == "random":
        return random_rank(dataset)
    if method in ("bm25_title", "bm25_title_body"):
        field = "title" if method == "bm25_title" else "title_body"
        return [{"pr_number": s["pr_number"], "ranked": bm25_rank(s, field), "gold": s["gold_issue_numbers"]}
                for s in dataset]
    model = model_id(backbone)
    log_dir = os.path.join(LOGS, tag)
    done = {}
    if out and os.path.exists(out + ".partial"):          # resume an interrupted run
        from common import load_json
        done = {r["pr_number"]: r for r in load_json(out + ".partial")}
    todo = [s for s in dataset if s["pr_number"] not in done]

    def work(s):
        t0 = time.time()
        try:
            r = rank_one(s, method, model, top_n, chunk_size, k, log_dir)
        except Exception as e:  # noqa: BLE001 - a failed sample falls back to BM25
            print(f"  PR#{s['pr_number']} failed: {e}")
            r = {"pr_number": s["pr_number"], "gold": s["gold_issue_numbers"], "ranked": bm25_rank(s)}
        r["elapsed"] = round(time.time() - t0, 1)
        return r

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, fut in enumerate(as_completed([ex.submit(work, s) for s in todo]), 1):
            r = fut.result()
            done[r["pr_number"]] = r
            hit = "HIT" if set(r["gold"]) & set(r["ranked"][:1]) else "miss"
            print(f"[{tag}] {i}/{len(todo)} PR#{r['pr_number']} {hit} ({r['elapsed']}s)")
            if out:
                save_json(list(done.values()), out + ".partial")
    return [done[s["pr_number"]] for s in dataset]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="pytorch, keras, ..., or design_study_pytorch")
    ap.add_argument("--method", default="progressive",
                    choices=["progressive", "bm25_then_llm", "single_title", "single_full",
                             "bm25_title", "bm25_title_body", "random"])
    ap.add_argument("--backbone", default="claude", help="kimi, minimax or claude (see CMG_MODEL_*)")
    ap.add_argument("--top_n", type=int, default=TOP_N, help="N: Stage-1 survivors")
    ap.add_argument("--chunk_size", type=int, default=CHUNK_SIZE)
    ap.add_argument("--k", type=int, default=100, help="K for single-stage methods (paper: 100 title, 30 full)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="only the first N samples")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dataset = load_samples(args.dataset, args.limit)
    llm_free = args.method in ("random", "bm25_title", "bm25_title_body")
    suffix = {"progressive": f"_N{args.top_n}", "bm25_then_llm": f"_N{args.top_n}",
              "single_title": f"_K{args.k}", "single_full": f"_K{args.k}"}.get(args.method, "")
    name = (args.method if llm_free else f"{args.backbone}_{args.method}") + suffix
    out = args.out or os.path.join(RESULTS, "runs", args.dataset, name + ".json")
    per_sample = run(dataset, args.method, None if llm_free else args.backbone, args.top_n,
                     args.chunk_size, args.k, args.workers, out, f"{args.dataset}_{name}")
    m = metrics(per_sample)
    save_json({"dataset": args.dataset, "backbone": None if llm_free else args.backbone, "method": name,
               "metrics": m, "per_sample": per_sample}, out)
    if os.path.exists(out + ".partial"):
        os.remove(out + ".partial")
    print({k: v for k, v in m.items()}, "->", out)


if __name__ == "__main__":
    main()
