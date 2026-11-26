"""Minimal GitHub REST client with token rotation and retry.

Tokens come from the environment, never from the code:

    export GITHUB_TOKENS=ghp_aaa,ghp_bbb     # several, rotated on rate limit
    export GITHUB_TOKEN=ghp_aaa              # or just one

Without a token GitHub allows 60 requests per hour, which is not enough to mine a
repository.
"""
import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://api.github.com"

_TOKENS = [t.strip() for t in (os.environ.get("GITHUB_TOKENS") or os.environ.get("GITHUB_TOKEN") or "").split(",") if t.strip()]
_reset_at = {t: 0.0 for t in _TOKENS}
_lock = threading.Lock()
_next = 0


def _pick_token():
    """A token that is not rate-limited right now; sleeps until one is free."""
    global _next
    if not _TOKENS:
        return None
    while True:
        with _lock:
            now = time.time()
            for _ in range(len(_TOKENS)):
                tok = _TOKENS[_next % len(_TOKENS)]
                _next += 1
                if _reset_at[tok] <= now:
                    return tok
            wait = max(2.0, min(_reset_at.values()) - now + 1)
        print(f"  [github] all tokens rate-limited, sleeping {wait:.0f}s")
        time.sleep(wait)


def _request(url, params=None, accept="application/vnd.github+json"):
    if params:
        url += "?" + urllib.parse.urlencode(params)
    last = None
    for attempt in range(8):
        tok = _pick_token()
        headers = {"Accept": accept, "User-Agent": "reason-matters-replication"}
        if tok:
            headers["Authorization"] = f"token {tok}"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            last = e
            h = dict(e.headers or {})
            remaining = h.get("X-RateLimit-Remaining") or h.get("x-ratelimit-remaining")
            if e.code in (403, 429) and tok and (remaining == "0" or e.code == 429):
                reset = h.get("X-RateLimit-Reset") or h.get("x-ratelimit-reset")
                with _lock:
                    _reset_at[tok] = max(_reset_at[tok], float(reset) if reset else time.time() + 60)
                continue
            if e.code in (500, 502, 503, 504) and attempt < 3:
                time.sleep(min(15, 2 ** attempt * 2))
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            last = e
            time.sleep(min(30, 2 ** attempt * 2))
    raise last or RuntimeError("GitHub request failed")


def get(path_or_url, params=None):
    """GET a JSON resource. `path_or_url` may be '/repos/o/r/issues' or a full URL."""
    url = path_or_url if path_or_url.startswith("http") else API + path_or_url
    return json.loads(_request(url, params).decode())


def get_diff(repo, pr_number):
    """The unified diff of a pull request."""
    return _request(f"{API}/repos/{repo}/pulls/{pr_number}", accept="application/vnd.github.v3.diff").decode(errors="replace")
