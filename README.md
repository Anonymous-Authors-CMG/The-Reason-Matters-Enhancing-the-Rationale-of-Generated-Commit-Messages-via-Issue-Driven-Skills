# Reason Matters: Enhancing the Rationale of Generated Commit Messages via Issue-Driven Skills

Replication package for the TOSEM submission.

📋 **Developer survey (RQ1)**
- Questionnaire: [Generating Commit Messages with LLMs: Your Voice Matters!](https://docs.google.com/forms/d/1MTgG3pYQbOPIzw46vhYYSqZ6hCk25AQ2FgBUYoIPgxg/edit?hl=zh-cn&pli=1#responses)
- All 59 responses: [exported spreadsheet](https://docs.google.com/spreadsheets/d/1E6DEMN9bK9JYrCKJops0oA8tIGQiF_waI5iEPb-iK64/edit?resourcekey=&gid=742278307#gid=742278307); a copy is in [`data/survey/responses.csv`](data/survey/responses.csv)

LLMs write a good summary of *what* a commit changes, but the *why* usually lives
in an issue nobody linked. This package contains everything behind the paper: the
1,200-sample evaluation set mined from six repositories, the Progressive Disclosure
retrieval that finds the issue a diff addresses, the commit messages generated with
and without that context, their What/Why scores from two judges, and a script that
recomputes every table from these files.

**Reproduce the paper's tables in one command, no API key needed:**

```bash
python src/report.py --check
```

---

- [The idea in one picture](#the-idea-in-one-picture)
- [Repository layout](#repository-layout)
- [Results](#results)
- [Data](#data)
- [Method and hyperparameters](#method-and-hyperparameters)
- [Running the pipeline](#running-the-pipeline)
- [Result file formats](#result-file-formats)
- [Developer survey and ethics](#developer-survey-and-ethics)
- [Tools built on this method](#tools-built-on-this-method)

## The idea in one picture

```
                 ┌──────────────────── Issue identification (Progressive Disclosure) ───────────────────┐
 staged diff ──▶ │ Stage 1: diff + issue TITLES, 120 per call ──▶ top N=10 ──▶ Stage 2: diff + title+BODY │ ──▶ top-1 issue ─┐
 open issues ──▶ │          (≤1,000 issues open at commit time)                of the 10 survivors       │                  │
                 └──────────────────────────────────────────────────────────────────────────────────────┘                  ▼
                 ┌──────────── RAG ────────────┐                                                               ┌──────────────────┐
 staged diff ──▶ │ top-5 similar historical    │ ────────────────────────────────────────────────────────────▶ │ generation prompt│ ──▶ commit message
                 │ commits (What + Why only)   │                                                               │ (role, 10 types) │     (What + Why)
                 └─────────────────────────────┘                                                               └──────────────────┘
```

Developers skim issue titles before opening the few that look right. Stage 1 does the
same, which lets a pool of 1,000 open issues fit in a handful of short LLM calls; only
ten issue bodies are ever read in full. No model is trained.

## Repository layout

```
.
├── src/                       all code, one script per step (see "Running the pipeline")
│   ├── common.py              paths, LLM backend, JSON parsing, dataset loader
│   ├── github_api.py          GitHub REST client (token rotation, retries)
│   ├── mine_links.py          step 1  mine issue-linked merged PRs and all issues
│   ├── build_dataset.py       step 2  candidate pools + 200 samples per repository
│   ├── identify_issues.py     step 3  Progressive Disclosure and all baselines     (RQ2)
│   ├── rag_retrieval.py       step 4  top-5 similar historical commits
│   ├── generate_messages.py   step 5  messages for the four context variants      (RQ3/RQ4)
│   ├── judge_messages.py      step 6  What/Why scores, 0–4
│   └── report.py              recompute every table; --check compares with the paper
├── prompts/                   every prompt, exactly as used
├── data/
│   ├── <repo>/                evaluation set of each repository (6 × 200 samples)
│   ├── design_study_pytorch/  PyTorch subset for Table 2 and Fig. 2 (pool of 300)
│   ├── extra/<repo>/          more linked samples, disjoint from the 200 (extended experiment)
│   ├── rag/                   RAG corpus and the retrieved exemplars
│   └── survey/responses.csv   the 59 survey responses (RQ1)
├── results/
│   ├── issue_identification/<repo>/<backbone>.json     Table 1
│   ├── design_study_pytorch/<method>.json               Table 2, Fig. 2
│   ├── commit_messages/<variant>/<repo>.json            Tables 4 and 5
│   └── extended/extra_samples/<repo>/<backbone>.json    extended experiment (not in the paper)
└── assets/                    ethics approval (redacted)
```

Backbone names in file names: `kimi` = Kimi-K2.5, `minimax` = MiniMax-M2.7,
`claude` = Claude-Opus-4.6. Context variants: `diff_only`, `exemplars` (+Exem.),
`issue` (+Issue), `full` (ours).

## Results

All numbers below are produced by `python src/report.py` from the files in this
repository, including the survey percentages of Section 3.2. `--check` additionally
compares every cell with the paper.

### RQ2. Issue identification (Table 1)

Six repositories, 200 samples each, candidate pool ≤ 1,000 open issues, N = 10.

| Domain | Repository | Kimi-K2.5 MRR | Hit@1 | R@10 | MiniMax-M2.7 MRR | Hit@1 | R@10 | Claude-Opus-4.6 MRR | Hit@1 | R@10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Model training | PyTorch | .806 | .775 | .840 | .814 | .780 | .866 | .855 | .835 | .873 |
| | Keras | .968 | .955 | .985 | .903 | .880 | .920 | .952 | .930 | .975 |
| | DeepSpeed | .843 | .800 | .910 | .816 | .775 | .888 | .862 | .830 | .928 |
| Storage | ClickHouse | .929 | .915 | .950 | .916 | .910 | .920 | .920 | .895 | .945 |
| | Pulsar | .928 | .895 | .980 | .908 | .865 | .954 | .944 | .920 | .968 |
| | RocketMQ | .850 | .825 | .875 | .882 | .860 | .915 | .947 | .930 | .965 |
| **6-repo weighted avg** | | **.887** | **.861** | **.923** | **.873** | **.845** | **.910** | **.913** | **.890** | **.942** |

With Claude-Opus-4.6 the correct issue is ranked first for 89% of commits and is in
the top 10 for 94%. `report.py` also prints Recall@1/5/20 for every cell.

### RQ2. Design choices (Table 2) and the choice of N (Fig. 2)

PyTorch design-study subset (`data/design_study_pytorch`, a pool of 300 candidates
per sample), Claude-Opus-4.6.

| Method | MRR | Hit@1 | R@10 |
|---|---:|---:|---:|
| Random | .032 | .000 | .059 |
| BM25 (title) | .308 | .217 | .548 |
| BM25 (title+body) | .457 | .377 | .638 |
| BM25 + LLM fine (N=10) | .634 | .623 | .638 |
| Single-stage (title, K=100) | .806 | .783 | .843 |
| Single-stage (full, K=30) | .702 | .696 | .710 |
| **Progressive (N=10)** | **.875** | **.855** | **.903** |

| Top-N | MRR | Hit@1 | R@5 | R@10 | R@20 |
|---|---:|---:|---:|---:|---:|
| N = 5 | .871 | .855 | .859 | .888 | .917 |
| **N = 10** | **.875** | **.855** | .879 | **.903** | **.932** |
| N = 20 | .860 | .841 | .886 | .896 | .914 |

Replacing BM25 with LLM title scanning in the first stage is what matters: the gold
issue is often not in BM25's top 30 at all. The same subset was also run with
Kimi-K2.5 and MiniMax-M2.7; `report.py` prints those rows too.

### RQ4. Context ablation (Table 4)

What/Why scores (0–4) pooled over the three generators, scored by two judges.

| Judge | Dimension | Diff-Only | +Exem. | +Issue | Full |
|---|---|---:|---:|---:|---:|
| DeepSeek-V4-Pro | What | 3.780 | 3.903 | 3.918 | **3.921** |
| DeepSeek-V4-Pro | Why | 2.547 | 3.752 | **3.931** | 3.906 |
| GPT-5.4 | What | **3.856** | 3.840 | 3.809 | 3.822 |
| GPT-5.4 | Why | 2.909 | 3.757 | 3.908 | **3.909** |

### RQ4. Why per generator (Table 5)

| Judge | Generator | Diff | +Exem. | +Issue | Full | Δ |
|---|---|---:|---:|---:|---:|---:|
| DeepSeek-V4-Pro | Kimi-K2.5 | 2.62 | 3.73 | 3.89 | **3.90** | +1.28 |
| | MiniMax-M2.7 | 2.37 | 3.57 | **3.91** | 3.83 | +1.54 |
| | Claude-Opus-4.6 | 2.65 | 3.97 | **4.00** | 3.99 | +1.35 |
| GPT-5.4 | Kimi-K2.5 | 2.94 | 3.71 | 3.87 | **3.89** | +0.95 |
| | MiniMax-M2.7 | 2.77 | 3.61 | **3.87** | 3.86 | +1.10 |
| | Claude-Opus-4.6 | 3.02 | 3.95 | 3.98 | **3.99** | +0.97 |

Δ is the best enriched variant minus Diff-Only. What stays near the ceiling in every
variant; Why is where context helps, and the issue is the largest single gain. Once
the issue is present, adding exemplars brings nothing further.

### Extended experiment: more samples per repository

The paper uses 200 linked samples per repository. To see how stable the
identification results are, we mined further linked samples that do not overlap
with those 200 (`data/extra/`) and ran the same Progressive N=10 pipeline on the
model-training repositories. DeepSpeed has only 297 valid linked samples in total,
so only 97 extra were available.

| Repository | Backbone | n | MRR | Hit@1 | R@10 | MRR on the main 200 |
|---|---|---:|---:|---:|---:|---:|
| PyTorch | Kimi-K2.5 | 190 | .897 | .863 | .937 | .806 |
| | MiniMax-M2.7 | 190 | .881 | .842 | .937 | .814 |
| | Claude-Opus-4.6 | 190 | .919 | .895 | .963 | .855 |
| Keras | Kimi-K2.5 | 190 | .884 | .853 | .926 | .968 |
| | MiniMax-M2.7 | 190 | .895 | .868 | .947 | .903 |
| | Claude-Opus-4.6 | 190 | .882 | .863 | .916 | .952 |
| DeepSpeed | Kimi-K2.5 | 97 | .627 | .577 | .691 | .843 |
| | MiniMax-M2.7 | 97 | .638 | .608 | .701 | .816 |
| | Claude-Opus-4.6 | 97 | .628 | .577 | .711 | .862 |

Two things move the numbers. The extra PyTorch samples come mostly from 2017–2018,
when fewer issues were open (median pool 342 instead of 1,000), so ranking is
easier. The extra DeepSpeed samples are linked almost only through Issue Events
(a commit that *mentions* an issue) rather than an explicit `Fixes #N`, and those
links are harder to recover from the diff. The extra samples for ClickHouse, Pulsar
and RocketMQ (250 each) are released but were not run.

## Data

### Evaluation sets

| Repository | Domain | GitHub | Samples | Avg pool | Median pool | Max pool | Commit dates |
|---|---|---|---:|---:|---:|---:|---|
| PyTorch | Model training | [pytorch/pytorch](https://github.com/pytorch/pytorch) | 200 | 1000 | 1000 | 1000 | 2020-10 – 2026-06 |
| Keras | Model training | [keras-team/keras](https://github.com/keras-team/keras) | 200 | 351 | 213 | 1000 | 2016-02 – 2026-05 |
| DeepSpeed | Model training | [deepspeedai/DeepSpeed](https://github.com/deepspeedai/DeepSpeed) | 200 | 732 | 770 | 1000 | 2020-04 – 2026-06 |
| ClickHouse | Storage infrastructure | [ClickHouse/ClickHouse](https://github.com/ClickHouse/ClickHouse) | 200 | 1000 | 1000 | 1000 | 2022-01 – 2026-06 |
| Pulsar | Storage infrastructure | [apache/pulsar](https://github.com/apache/pulsar) | 200 | 779 | 940 | 1000 | 2018-01 – 2026-06 |
| RocketMQ | Storage infrastructure | [apache/rocketmq](https://github.com/apache/rocketmq) | 200 | 250 | 252 | 457 | 2022-08 – 2026-04 |

Each sample is a merged pull request linked to at least one issue, with:

- **the diff** of the PR, as GitHub serves it;
- **the gold issue(s)** it fixes;
- **the candidate pool**: the issues that were open when the commit landed, i.e.
  created before it and not yet closed, capped at 1,000 (the gold issue plus the
  most recently created others). Only titles, bodies (≤ 2,500 characters) and labels
  are kept; nothing from the future is visible.

**How links were mined** (`src/mine_links.py`). Two signals, both recorded by GitHub
itself: (1) `Fixes/Closes/Resolves #N` in the PR body or title, in any tense and
case; (2) the issue's `referenced` and `closed` events that carry a commit id,
resolved to the merged PR containing that commit. A manual check of 100 random links
found more than 95% correct.

**Why 200 per repository.** DeepSpeed, the smallest repository, has only 297 valid
linked samples after scanning every closed issue and commit reference, so 200 is
close to its whole population. With Cochran's formula and finite population
correction, 200 samples give a margin of error ≤ 5.9% at 95% confidence for any of
the six. Larger repositories were sampled evenly over commit time.

TensorFlow was mined too but yields only 136 valid linked samples, so it was left out.

**Format.** Each set is two gzipped JSON-lines files. Issues that appear in many pools
are stored once:

```
data/<repo>/samples.jsonl.gz   pr_number, pr_title, commit_datetime, gold_issue_numbers,
                               candidate_issue_numbers, diff_text
data/<repo>/issues.jsonl.gz    number, title, body, labels, created_at, closed_at
```

```python
import sys; sys.path.insert(0, "src")
from common import load_samples

samples = load_samples("keras")              # list of 200 dicts
s = samples[0]
s["diff_text"], s["gold_issue_numbers"], len(s["candidate_pool"])   # pool: [{number, title, body, labels}, ...]
```

### RAG corpus and retrieved exemplars

`data/rag/corpus.jsonl.gz` holds 4,273 historical commits from open-source projects
in the two studied domains, model training and storage infrastructure, each with its
diff, message and linked issue titles. Exemplars are retrieved from the same domain
as the query. `data/rag/top5_<repo>.json` lists the five exemplars retrieved for each
sample, with their similarity; these are exactly the exemplars used in the
`exemplars` and `full` variants.

## Method and hyperparameters

| Component | Setting |
|---|---|
| **Candidate pool** | issues open at commit time, ≤ 1,000 (gold + most recent), ≥ 20 |
| **Stage 1** (titles) | 120 titles per call; each chunk keeps ⌈1.5·N⌉; one merge pass keeps N |
| **Stage 2** (full text) | title + body of the N = 10 survivors; ranks all ten |
| Ranking completion | issues not reached by Stage 2 follow in BM25 (title + body) order |
| Prompt budget, identification | diff 8,000 chars, title 200, body 1,500 |
| Output parsing | JSON array → first `[...]` → all integers; up to 2 retries with a stricter reminder |
| **RAG** | top-5 by cosine similarity; near-duplicates (≥ 0.999) skipped; domain corpus |
| **Generation** | diff ≤ 24,000 chars; 10 commit types (Zeng et al.); sampling via `--temperature` / `--top_p` |
| **Judges** | DeepSeek-V4-Pro and GPT-5.4, CMEval few-shot CoT rubric, What and Why on 0–4 |
| Backbones | Kimi-K2.5, MiniMax-M2.7, Claude-Opus-4.6; neither judge is used as a generator |

The prompts are in `prompts/`:

| File | Used for |
|---|---|
| `stage1_title_ranking.txt` | Stage 1, and the single-stage title baseline |
| `stage2_full_ranking.txt` | Stage 2, BM25 + LLM, and the single-stage full-text baseline |
| `json_retry_reminder.txt` | appended when a reply has no usable issue number |
| `generate_diff_only.txt` | Diff-Only |
| `generate_exemplars.txt` / `generate_issue.txt` / `generate_full.txt` | +Exem. / +Issue / Full |
| `commit_types.txt` | the ten commit types inserted into the three enriched prompts |
| `judge_what_why.txt` | the CMEval What/Why rubric |

## Running the pipeline

Python 3.9+. Every step except RAG retrieval uses only the standard library.

### Configure a model backend

The scripts talk to models in one of two ways:

```bash
# an OpenAI-compatible API
export CMG_API_KEY=sk-...
export CMG_BASE_URL=https://api.openai.com/v1      # or any compatible endpoint

# or a command-line agent, one call per prompt ({prompt} and {model} are filled in)
export CMG_LLM_CMD="claude -p {prompt} --model {model}"
```

Backbone short names map to model ids through `CMG_MODEL_<NAME>`, for example:

```bash
export CMG_MODEL_CLAUDE=claude-opus-4-6
export CMG_MODEL_KIMI=kimi-k2.5
export CMG_MODEL_MINIMAX=MiniMax-M2.7
```

New runs are written under `results/runs/`, so the published results are never
overwritten. Raw prompts and replies go to `logs/`. Long runs can be resumed by
running the same command again.

### Step 1–2. Mine links and build the evaluation sets (optional)

The evaluation sets are already in `data/`. To rebuild one from GitHub:

```bash
export GITHUB_TOKENS=ghp_aaa,ghp_bbb             # several tokens are rotated on rate limits
python src/mine_links.py issues deepspeed        # every issue (listing + monthly search)
python src/mine_links.py prs deepspeed           # PRs with Fixes/Closes/Resolves #N, with diffs
python src/mine_links.py events deepspeed        # more links from the Issue Events API
python src/build_dataset.py deepspeed --out data/deepspeed_rebuilt
```

Raw data goes to `raw/` (git-ignored; several GB for the large repositories). GitHub
keeps changing, so a fresh mining run will not return exactly the same PRs.

### Step 3. Issue identification

```bash
# Table 1, one repository and backbone
python src/identify_issues.py --dataset pytorch --backbone claude

# Table 2 and Fig. 2
python src/identify_issues.py --dataset design_study_pytorch --method random
python src/identify_issues.py --dataset design_study_pytorch --method bm25_title
python src/identify_issues.py --dataset design_study_pytorch --method bm25_title_body
python src/identify_issues.py --dataset design_study_pytorch --method bm25_then_llm --backbone claude
python src/identify_issues.py --dataset design_study_pytorch --method single_title --k 100 --backbone claude
python src/identify_issues.py --dataset design_study_pytorch --method single_full --k 30 --backbone claude
python src/identify_issues.py --dataset design_study_pytorch --top_n 5 --backbone claude   # also 10, 20
```

The three non-LLM rows of Table 2 rerun bit-for-bit. For a quick check, add
`--limit 5`.

### Step 4. Retrieve exemplars

```bash
pip install -r requirements.txt
python src/rag_retrieval.py --repos pytorch keras deepspeed
```

### Step 5–6. Generate and score commit messages

```bash
for v in diff_only exemplars issue full; do
  python src/generate_messages.py --variant $v --temperature 0.8 --top_p 0.95
done
python src/judge_messages.py results/runs/commit_messages/*/*.json --judge_model deepseek-v4-pro --label deepseek-v4-pro
python src/judge_messages.py results/runs/commit_messages/*/*.json --judge_model gpt-5.4 --label GPT-5.4
```

`--issue_source identified` feeds the generator the top issue found in step 3
instead of the gold one.

## Result file formats

**`results/issue_identification/<repo>/<backbone>.json`** and
**`results/design_study_pytorch/<method>.json`**

```jsonc
{
  "repo": "keras", "backbone": "claude", "method": "progressive", "top_n": 10, ...,
  "metrics": {"n": 200, "MRR": 0.9524, "Hit@1": 0.93, "Recall@1": ..., "Recall@20": ...},
  "per_sample": [
    {"pr_number": 1593, "gold": [1567],
     "stage1_topN": [1567, ...],          // the 10 issues that reached Stage 2
     "ranked": [1567, ...],               // full ranking of the pool
     "elapsed": 85.9}                     // seconds
  ]
}
```

**`results/commit_messages/<variant>/<repo>.json`**, one entry per sample:

```jsonc
{
  "repo": "keras", "pr_number": 1593, "pr_title": "...", "commit_datetime": "...",
  "gold_issue_numbers": [1567], "diff_chars": 9480, "diff_truncated": false,
  "context": {"exemplars": [...], "issue": "Issue #1567: ..."},   // what the prompt contained
  "messages": {"kimi": "...", "minimax": "...", "claude": "..."},
  "scores": {
    "deepseek-v4-pro": {"what": {"kimi": 4, ...}, "why": {"kimi": 4, ...}},
    "GPT-5.4":         {"what": {...},             "why": {...}}
  }
}
```

A score of `null` means the judge's reply could not be parsed; such messages are left
out of the averages.

## Developer survey and ethics

The motivating survey (RQ1, 59 valid responses) used this questionnaire:
[Generating Commit Messages with LLMs: Your Voice Matters!](https://docs.google.com/forms/d/1MTgG3pYQbOPIzw46vhYYSqZ6hCk25AQ2FgBUYoIPgxg/edit?hl=zh-cn&pli=1#responses)
(now closed). The responses are in the [exported spreadsheet](https://docs.google.com/spreadsheets/d/1E6DEMN9bK9JYrCKJops0oA8tIGQiF_waI5iEPb-iK64/edit?resourcekey=&gid=742278307#gid=742278307) and in
`data/survey/responses.csv` (timestamps converted to ISO format; no personal data is
collected). `python src/report.py` recomputes the percentages reported in Section 3.2.

Participants were contacted only if they had made their email address public on
their GitHub profile, in line with
[GitHub's Acceptable Use Policies](https://docs.github.com/en/site-policy/acceptable-use-policies).
The study was approved by our institution's ethics committee
([English](assets/ethics_review_en.png), [Chinese](assets/ethics_review_cn.png);
identifying details are redacted for double-blind review).

## Tools built on this method

The same method, packaged for daily use:
[CMG-Skills](https://github.com/Anonymous-Authors-CMG/CMG-Skills) provides an agent
skill (Claude Code, Codex, Cursor, …), a VS Code extension, and a GitHub Action /
GitLab CI bot that suggests the linked issue on every pull request.

## Citation

```bibtex
@article{reasonmatters2026,
  title   = {Reason Matters: Enhancing the Rationale of Generated Commit Messages via Issue-Driven Skills},
  author  = {Anonymous Authors},
  journal = {ACM Transactions on Software Engineering and Methodology},
  note    = {Under review},
  year    = {2026}
}
```
