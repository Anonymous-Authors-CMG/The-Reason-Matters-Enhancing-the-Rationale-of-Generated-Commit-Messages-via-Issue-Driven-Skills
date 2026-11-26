"""Recompute every table in the paper from the files in results/.

  python src/report.py                 # print all tables as Markdown
  python src/report.py --check         # also compare against the numbers printed in the paper

Tables: survey (RQ1), dataset statistics, Table 1 (issue identification, 6 repos x 3 backbones),
Table 2 (design choices on the PyTorch subset), Fig. 2 (sensitivity to N), Table 4
(context ablation, pooled), Table 5 (per-backbone Why), and the extended experiment
on additional samples.
"""
import argparse
import csv
import glob
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BACKBONES, DATA, DOMAINS, REPOS, RESULTS, load_json, load_samples  # noqa: E402

NAMES = {"kimi": "Kimi-K2.5", "minimax": "MiniMax-M2.7", "claude": "Claude-Opus-4.6"}
REPO_NAMES = {"pytorch": "PyTorch", "keras": "Keras", "deepspeed": "DeepSpeed",
              "clickhouse": "ClickHouse", "pulsar": "Pulsar", "rocketmq": "RocketMQ"}
JUDGES = ["deepseek-v4-pro", "GPT-5.4"]
JUDGE_NAMES = {"deepseek-v4-pro": "DeepSeek-V4-Pro", "GPT-5.4": "GPT-5.4"}
VARIANTS = [("diff_only", "Diff-Only"), ("exemplars", "+Exem."), ("issue", "+Issue"), ("full", "Full")]

# Numbers as printed in the paper (Section 3.2 and Tables 1, 2, 4, 5), used by --check.
PAPER = {
    "survey": {"n": 59, "experience_over_3y": 71.2, "tried_ai_commit_messages": 67.8, "importance_4_or_5": 86.4,
               "satisfaction_mean": 3.15, "q6_summarize": 83.1, "q6_type": 71.2, "q6_reasons": 55.9,
               "q7_issue_references": 49.2, "q7_reasons": 45.8,
               # 46/59 = 77.97%; the paper prints 77.9%
               "ai_assistant_always_or_frequently": 78.0},
    "table1": {  # repo -> backbone -> (MRR, Hit@1, R@10)
        "pytorch": {"kimi": (.806, .775, .840), "minimax": (.814, .780, .866), "claude": (.855, .835, .873)},
        "keras": {"kimi": (.968, .955, .985), "minimax": (.903, .880, .920), "claude": (.952, .930, .975)},
        "deepspeed": {"kimi": (.843, .800, .910), "minimax": (.816, .775, .888), "claude": (.862, .830, .928)},
        "clickhouse": {"kimi": (.929, .915, .950), "minimax": (.916, .910, .920), "claude": (.920, .895, .945)},
        "pulsar": {"kimi": (.928, .895, .980), "minimax": (.908, .865, .954), "claude": (.944, .920, .968)},
        "rocketmq": {"kimi": (.850, .825, .875), "minimax": (.882, .860, .915), "claude": (.947, .930, .965)},
        "avg": {"kimi": (.887, .861, .923), "minimax": (.873, .845, .910), "claude": (.913, .890, .942)},
    },
    "table2": {"random": (.032, .000, .059), "bm25_title": (.308, .217, .548),
               "bm25_title_body": (.457, .377, .638), "claude_bm25_then_llm_N10": (.634, .623, .638),
               "claude_single_stage_title_K100": (.806, .783, .843),
               "claude_single_stage_full_K30": (.702, .696, .710), "claude_progressive_N10": (.875, .855, .903)},
    "table4": {"deepseek-v4-pro": {"what": (3.780, 3.903, 3.918, 3.921), "why": (2.547, 3.752, 3.931, 3.906)},
               "GPT-5.4": {"what": (3.856, 3.840, 3.809, 3.822), "why": (2.909, 3.757, 3.908, 3.909)}},
    "table5": {"deepseek-v4-pro": {"kimi": (2.620, 3.730, 3.890, 3.900), "minimax": (2.370, 3.570, 3.910, 3.830),
                                   "claude": (2.650, 3.970, 4.000, 3.990)},
               # +Issue for Kimi under GPT-5.4: the exact mean is 3.8748, so 3.87 (printed as 3.880 in the submitted draft)
               "GPT-5.4": {"kimi": (2.940, 3.710, 3.870, 3.890), "minimax": (2.770, 3.610, 3.870, 3.860),
                           "claude": (3.020, 3.950, 3.980, 3.990)}},
}

mismatches = []


def check(label, got, want, tol):
    if abs(got - want) > tol:
        mismatches.append(f"{label}: computed {got:.4f}, paper {want}")


def f3(x):
    """Three decimals, rounded half-up (as in the paper)."""
    return str(Decimal(repr(x)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" if i == 0 else "---:" for i in range(len(headers))) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def dataset_stats():
    rows = []
    for domain, repos in DOMAINS.items():
        for r in repos:
            s = load_samples(r)
            pools = sorted(len(x["candidate_pool"]) for x in s)
            dates = sorted(x["commit_datetime"][:10] for x in s)
            rows.append([REPO_NAMES[r], domain, REPOS[r], len(s), round(sum(pools) / len(pools)), pools[0],
                         pools[len(pools) // 2], pools[-1], f"{dates[0]} – {dates[-1]}"])
    return table(["Repository", "Domain", "GitHub", "Samples", "Avg pool", "Min", "Median", "Max", "Commit dates"], rows)


def table1(do_check):
    res = {(r, b): load_json(os.path.join(RESULTS, "issue_identification", r, f"{b}.json"))["metrics"]
           for r in REPOS for b in BACKBONES}
    rows = []
    for domain, repos in DOMAINS.items():
        for r in repos:
            row = [domain if r == repos[0] else "", REPO_NAMES[r]]
            for b in BACKBONES:
                m = res[(r, b)]
                row += [f3(m["MRR"]), f3(m["Hit@1"]), f3(m["Recall@10"])]
                if do_check:
                    for i, k in enumerate(("MRR", "Hit@1", "Recall@10")):
                        check(f"Table 1 {r}/{b}/{k}", m[k], PAPER["table1"][r][b][i], 0.0006)
            rows.append(row)
    avg = ["**6-repo weighted avg**", ""]
    for b in BACKBONES:
        n = sum(res[(r, b)]["n"] for r in REPOS)
        for i, k in enumerate(("MRR", "Hit@1", "Recall@10")):
            v = sum(res[(r, b)][k] * res[(r, b)]["n"] for r in REPOS) / n
            avg.append(f"**{f3(v)}**")
            if do_check:
                check(f"Table 1 avg/{b}/{k}", v, PAPER["table1"]["avg"][b][i], 0.0006)
    rows.append(avg)
    headers = ["Domain", "Repository"] + [f"{NAMES[b]} {k}" for b in BACKBONES for k in ("MRR", "Hit@1", "R@10")]
    full = []
    for b in BACKBONES:
        for r in REPOS:
            m = res[(r, b)]
            full.append([NAMES[b], REPO_NAMES[r], m["n"]] + [f3(m[k]) for k in
                        ("MRR", "Hit@1", "Recall@1", "Recall@5", "Recall@10", "Recall@20")])
    return (table(headers, rows),
            table(["Backbone", "Repository", "n", "MRR", "Hit@1", "R@1", "R@5", "R@10", "R@20"], full))


def table2(do_check):
    d = os.path.join(RESULTS, "design_study_pytorch")
    order = [("random", "Random"), ("bm25_title", "BM25 (title)"), ("bm25_title_body", "BM25 (title+body)"),
             ("claude_bm25_then_llm_N10", "BM25 + LLM fine (N=10)"),
             ("claude_single_stage_title_K100", "Single-stage (title, K=100)"),
             ("claude_single_stage_full_K30", "Single-stage (full, K=30)"),
             ("claude_progressive_N10", "Progressive (N=10)")]
    rows = []
    for key, name in order:
        m = load_json(os.path.join(d, key + ".json"))["metrics"]
        rows.append([name, m["n"], f3(m["MRR"]), f3(m["Hit@1"]), f3(m["Recall@10"])])
        if do_check:
            for i, k in enumerate(("MRR", "Hit@1", "Recall@10")):
                check(f"Table 2 {key}/{k}", m[k], PAPER["table2"][key][i], 0.0006)
    sens = []
    for n in (5, 10, 20):
        m = load_json(os.path.join(d, f"claude_progressive_N{n}.json"))["metrics"]
        sens.append([f"N={n}", f3(m["MRR"]), f3(m["Hit@1"]), f3(m["Recall@5"]), f3(m["Recall@10"]), f3(m["Recall@20"])])
    other = []
    for b in ("kimi", "minimax"):
        for f in sorted(glob.glob(os.path.join(d, f"{b}_*.json"))):
            m = load_json(f)["metrics"]
            other.append([NAMES[b], os.path.basename(f)[len(b) + 1:-5], f3(m["MRR"]), f3(m["Hit@1"]), f3(m["Recall@10"])])
    return (table(["Method", "n", "MRR", "Hit@1", "R@10"], rows),
            table(["Top-N", "MRR", "Hit@1", "R@5", "R@10", "R@20"], sens),
            table(["Backbone", "Method", "MRR", "Hit@1", "R@10"], other))


def load_scores():
    """scores[variant][judge][dim][backbone] -> list of ints, pooled over repos."""
    scores = {}
    for v, _ in VARIANTS:
        for f in sorted(glob.glob(os.path.join(RESULTS, "commit_messages", v, "*.json"))):
            for rec in load_json(f):
                for judge, sc in rec["scores"].items():
                    for dim in ("what", "why"):
                        for b, x in sc[dim].items():
                            if isinstance(x, int):
                                scores.setdefault(v, {}).setdefault(judge, {}).setdefault(dim, {}).setdefault(b, []).append(x)
    return scores


def mean(xs):
    return sum(xs) / len(xs)


def r2(xs):
    """Exact mean rounded half-up to 2 decimals (how Table 5 is printed)."""
    f = Fraction(sum(xs), len(xs))
    return Decimal(f.numerator) / Decimal(f.denominator)


def two(d):
    return d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def table4_5(do_check):
    s = load_scores()
    rows4 = []
    for j in JUDGES:
        for dim in ("what", "why"):
            # pooled over every scored message of the three generators (as in the paper)
            vals = [float(r2([x for b in BACKBONES for x in s[v][j][dim][b]])) for v, _ in VARIANTS]
            rows4.append([JUDGE_NAMES[j], dim.capitalize()] + [f3(x) for x in vals])
            if do_check:
                for x, want, (v, _) in zip(vals, PAPER["table4"][j][dim], VARIANTS):
                    check(f"Table 4 {j}/{dim}/{v}", x, want, 0.0006)
    rows5 = []
    for j in JUDGES:
        for b in BACKBONES:
            vals = [two(r2(s[v][j]["why"][b])) for v, _ in VARIANTS]
            rows5.append([JUDGE_NAMES[j], NAMES[b]] + [str(x) for x in vals] + [f"+{max(vals[1:]) - vals[0]}"])
            if do_check:
                for x, want, (v, _) in zip(vals, PAPER["table5"][j][b], VARIANTS):
                    check(f"Table 5 {j}/{b}/{v}", float(x), want, 0.0001)
    n = {v: sum(len(s[v]["deepseek-v4-pro"]["why"][b]) for b in BACKBONES) for v, _ in VARIANTS}
    repos = sorted({os.path.basename(f)[:-5] for f in glob.glob(os.path.join(RESULTS, "commit_messages", "*", "*.json"))})
    note = f"Messages scored per variant: {n['full']} ({len(repos)} repos: {', '.join(REPO_NAMES[r] for r in repos)}; 3 backbones)."
    return (table(["Judge", "Dimension"] + [n for _, n in VARIANTS], rows4),
            table(["Judge", "Generator"] + [n for _, n in VARIANTS] + ["Δ"], rows5)
            + "\n\nTwo decimals (half-up), as in the paper; Δ = best enriched variant − Diff-Only.", note)


def survey(do_check):
    rows = list(csv.DictReader(open(os.path.join(DATA, "survey", "responses.csv"), encoding="utf-8")))
    cols = list(rows[0].keys())
    n = len(rows)

    def share(pred):
        return round(100 * sum(1 for r in rows if pred(r)) / n, 1)

    def picked(col, option):
        return share(lambda r: option in r[col])

    got = {
        "n": n,
        "experience_over_3y": share(lambda r: r[cols[1]] == "More than 3 years"),
        "importance_4_or_5": share(lambda r: r[cols[2]] in ("4", "5")),
        "ai_assistant_always_or_frequently": share(lambda r: r[cols[3]] in ("Always", "Frequently")),
        "tried_ai_commit_messages": share(lambda r: r[cols[4]] == "Yes"),
        "satisfaction_mean": round(sum(int(r[cols[5]]) for r in rows) / n, 2),
        "q6_summarize": picked(cols[6], "Summarize the changes"),
        "q6_type": picked(cols[6], "Type of commit"),
        "q6_reasons": picked(cols[6], "Describe the reasons"),
        "q7_issue_references": picked(cols[7], "References to related tasks or issues"),
        "q7_reasons": picked(cols[7], "Describe the reasons"),
    }
    labels = {
        "n": "Valid responses", "experience_over_3y": "More than 3 years of experience (%)",
        "ai_assistant_always_or_frequently": "Use AI assistants always or frequently (%)",
        "tried_ai_commit_messages": "Have tried LLM-generated commit messages (%)",
        "importance_4_or_5": "Rate commit messages 4 or 5 of 5 in importance (%)",
        "satisfaction_mean": "Satisfaction with LLM commit messages (mean, 1–5)",
        "q6_summarize": "Essential: summarize the changes (%)", "q6_type": "Essential: commit type (%)",
        "q6_reasons": "Essential: reasons for the changes (%)",
        "q7_issue_references": "Missing: references to related tasks or issues (%)",
        "q7_reasons": "Missing: reasons for the changes (%)",
    }
    if do_check:
        for k, v in got.items():
            check(f"Survey {k}", v, PAPER["survey"][k], 0.051)
    return table(["Finding", "Value"], [[labels[k], v] for k, v in got.items()])


def extended():
    rows = []
    for r in ("pytorch", "keras", "deepspeed"):
        for b in BACKBONES:
            m = load_json(os.path.join(RESULTS, "extended", "extra_samples", r, f"{b}.json"))["metrics"]
            main_m = load_json(os.path.join(RESULTS, "issue_identification", r, f"{b}.json"))["metrics"]
            rows.append([REPO_NAMES[r], NAMES[b], m["n"], f3(m["MRR"]), f3(m["Hit@1"]), f3(m["Recall@10"]),
                         f3(main_m["MRR"])])
    return table(["Repository", "Backbone", "n", "MRR", "Hit@1", "R@10", "MRR on the main 200"], rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="compare with the numbers printed in the paper")
    args = ap.parse_args()
    print("## RQ1: developer survey (data/survey/responses.csv)\n\n" + survey(args.check))
    print("\n## Evaluation sets\n\n" + dataset_stats())
    t1, t1_full = table1(args.check)
    print("\n## Table 1: issue identification (pool <= 1000, N = 10)\n\n" + t1)
    print("\n### All recall levels\n\n" + t1_full)
    t2, fig2, other = table2(args.check)
    print("\n## Table 2: design choices (PyTorch subset, pool = 300, Claude-Opus-4.6)\n\n" + t2)
    print("\n## Fig. 2: sensitivity to N\n\n" + fig2)
    print("\n### Same subset, other backbones\n\n" + other)
    t4, t5, note = table4_5(args.check)
    print("\n## Table 4: context ablation, pooled over generators\n\n" + t4 + "\n\n" + note)
    print("\n## Table 5: Why per generator\n\n" + t5)
    print("\n## Extended: progressive N=10 on additional samples (not in the paper)\n\n" + extended())
    if args.check:
        print("\n## Check against the paper\n")
        print("All values match." if not mismatches else "\n".join("- " + m for m in mismatches))
        sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
