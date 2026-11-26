"""Shared helpers: paths, the LLM backend, JSON parsing, and the dataset format.

Every other script imports from here, so this is the one place to change how the
models are called or where data lives.
"""
import gzip
import json
import os
import re
import shlex
import subprocess
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
PROMPTS = os.path.join(ROOT, "prompts")
LOGS = os.path.join(ROOT, "logs")          # raw prompts/outputs, git-ignored

# The six evaluation repositories (Section 4.1 of the paper).
REPOS = {
    "pytorch": "pytorch/pytorch",
    "keras": "keras-team/keras",
    "deepspeed": "deepspeedai/DeepSpeed",
    "clickhouse": "ClickHouse/ClickHouse",
    "pulsar": "apache/pulsar",
    "rocketmq": "apache/rocketmq",
}
DOMAINS = {
    "Model training": ["pytorch", "keras", "deepspeed"],
    "Storage infrastructure": ["clickhouse", "pulsar", "rocketmq"],
}

# Short names of the three backbone LLMs, as used in all result files.
BACKBONES = ["kimi", "minimax", "claude"]


def prompt(name):
    """Read a prompt template from prompts/ exactly as it was used."""
    with open(os.path.join(PROMPTS, name), encoding="utf-8") as f:
        return f.read()


def truncate_head_tail(text, limit, marker="\n...[DIFF TRUNCATED]...\n"):
    """Keep the first 70% and the last 30% of `limit` characters."""
    if not text or len(text) <= limit:
        return text or ""
    return text[: int(limit * 0.7)] + marker + text[-int(limit * 0.3):]


# --------------------------------------------------------------------- LLM backend
#
# Two ways to reach a model, chosen by environment variables:
#
#   OpenAI-compatible HTTP API          CMG_API_KEY, CMG_BASE_URL (default OpenAI)
#   Any command-line agent              CMG_LLM_CMD, e.g. "claude -p" or
#                                        "codebuddy -p {prompt} --model {model}"
#
# The experiments in the paper were run through a command-line agent, one call per
# prompt, with the model chosen by --model.

def _call_http(text, model, timeout, temperature, top_p):
    base = os.environ.get("CMG_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    url = base if base.endswith("/chat/completions") else base + "/chat/completions"
    body = {"model": model, "messages": [{"role": "user", "content": text}], "stream": False}
    if temperature is not None:
        body["temperature"] = temperature
    if top_p is not None:
        body["top_p"] = top_p
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST", headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer " + os.environ["CMG_API_KEY"],
    })
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())["choices"][0]["message"]["content"] or ""


def _call_cli(text, model, timeout):
    template = os.environ["CMG_LLM_CMD"]
    if "{prompt}" in template:
        argv = [a.replace("{model}", model or "") for a in shlex.split(template)]
        argv = [text if a == "{prompt}" else a for a in argv]
        argv = [a for a in argv if a != ""]
    else:
        argv = shlex.split(template) + [text]
        if model:
            argv += ["--model", model]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                          stdin=subprocess.DEVNULL)
    if proc.returncode != 0:
        raise RuntimeError(f"{argv[0]} exit {proc.returncode}: {proc.stderr[:300]}")
    return proc.stdout


def llm(text, model, timeout=600, retries=2, temperature=None, top_p=None):
    """Send one prompt and return the raw reply. Retries on any error."""
    if not (os.environ.get("CMG_API_KEY") or os.environ.get("CMG_LLM_CMD")):
        raise RuntimeError("Set CMG_API_KEY (+ CMG_BASE_URL) or CMG_LLM_CMD; see README.")
    last = None
    for attempt in range(retries + 1):
        try:
            if os.environ.get("CMG_API_KEY"):
                return _call_http(text, model, timeout, temperature, top_p)
            return _call_cli(text, model, timeout)
        except Exception as e:  # noqa: BLE001 - retry on anything
            last = e
            print(f"  llm call failed (attempt {attempt}): {str(e)[:200]}")
            time.sleep(3)
    raise last


def model_id(short):
    """Map a backbone short name to the model id sent to the backend.

    Override with e.g. CMG_MODEL_CLAUDE=claude-opus-4-6, CMG_MODEL_KIMI=kimi-k2.5.
    """
    return os.environ.get(f"CMG_MODEL_{short.upper()}", short)


def extract_json(text):
    """Return the first JSON array/object in a model reply (fenced or not), or None."""
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    cand = m.group(1) if m else text
    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        i = cand.find(open_ch)
        if i < 0:
            continue
        depth, in_str, esc = 0, False, False
        for j in range(i, len(cand)):
            ch = cand[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(cand[i:j + 1])
                    except Exception:
                        break
    try:
        return json.loads(cand)
    except Exception:
        return None


def extract_issue_numbers(text):
    """Three-layer parser for a ranked list of issue numbers (Section 7.1).

    1. a JSON array (fenced or bare);
    2. the first [...] of comma-separated numbers;
    3. every 2-7 digit integer in order of appearance.
    """
    if not text:
        return []
    parsed = extract_json(text)
    if isinstance(parsed, list):
        nums = [int(str(x).lstrip("#").strip()) for x in parsed
                if isinstance(x, (int, str)) and str(x).lstrip("#").strip().isdigit()]
        if nums:
            return nums
    m = re.search(r"\[\s*(?:#?\d+(?:\s*,\s*#?\d+)*)\s*\]", text)
    if m:
        nums = [int(t) for t in re.findall(r"#?(\d+)", m.group(0))]
        if nums:
            return nums
    seen, out = set(), []
    for x in re.findall(r"#?(\d{2,7})", text):
        n = int(x)
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


# --------------------------------------------------------------------- data format
#
# Each evaluation set is stored as two gzipped JSON-lines files:
#
#   samples.jsonl.gz   one merged PR per line: pr_number, pr_title, commit_datetime,
#                      diff_text, gold_issue_numbers, candidate_issue_numbers
#   issues.jsonl.gz    every issue that appears in any candidate pool, once
#
# load_samples() joins them back into the in-memory form the scripts use, where
# each sample has a `candidate_pool` list of {number, title, body, labels}.

POOL_FIELDS = ("number", "title", "body", "labels")


def _read_jsonl(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # mtime=0 keeps the archive byte-identical across runs
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        for r in rows:
            gz.write((json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8"))


def dataset_dir(name):
    """data/<name>/, e.g. data/pytorch or data/design_study_pytorch."""
    return os.path.join(DATA, name)


def save_samples(samples, out_dir, issue_meta=None):
    """Write samples in the compact format. `issue_meta` may add created_at/closed_at."""
    issues = {}
    rows = []
    for s in samples:
        rows.append({
            "pr_number": s["pr_number"],
            "pr_title": s.get("pr_title", ""),
            "commit_datetime": s["commit_datetime"],
            "gold_issue_numbers": s["gold_issue_numbers"],
            "candidate_issue_numbers": [c["number"] for c in s["candidate_pool"]],
            "diff_text": s["diff_text"],
        })
        for c in s["candidate_pool"]:
            item = {k: c.get(k) for k in POOL_FIELDS}
            prev = issues.get(c["number"])
            if prev is not None and prev != item:
                raise ValueError(f"issue #{c['number']} differs between pools")
            issues[c["number"]] = item
    table = []
    for n in sorted(issues):
        item = dict(issues[n])
        meta = (issue_meta or {}).get(n) or {}
        for k in ("created_at", "closed_at"):
            if k in meta:
                item[k] = meta[k]
        table.append(item)
    _write_jsonl(os.path.join(out_dir, "samples.jsonl.gz"), rows)
    _write_jsonl(os.path.join(out_dir, "issues.jsonl.gz"), table)
    return len(rows), len(table)


def load_samples(name_or_dir, limit=0):
    """Load an evaluation set (by name under data/, or a directory path)."""
    d = name_or_dir if os.path.isdir(name_or_dir) else dataset_dir(name_or_dir)
    issues = {i["number"]: {k: i.get(k) for k in POOL_FIELDS}
              for i in _read_jsonl(os.path.join(d, "issues.jsonl.gz"))}
    out = []
    for row in _read_jsonl(os.path.join(d, "samples.jsonl.gz")):
        s = {k: row[k] for k in ("pr_number", "pr_title", "commit_datetime", "diff_text",
                                 "gold_issue_numbers")}
        s["candidate_pool"] = [issues[n] for n in row["candidate_issue_numbers"]]
        out.append(s)
        if limit and len(out) >= limit:
            break
    return out


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path, indent=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
    os.replace(tmp, path)
