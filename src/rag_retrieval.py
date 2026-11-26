"""Step 4 - retrieve similar historical commits as exemplars (Section 4.2, RAG module).

The corpus (data/rag/corpus.jsonl.gz) holds 4,273 historical commits from projects
in the two studied domains, each with its diff and message.
Query diffs and corpus diffs are embedded with the same model, and for each query
the 5 most similar commits by cosine similarity are kept. Near-identical hits
(cosine >= 0.999, i.e. the query commit itself) are skipped.

Diffs longer than 12,000 characters keep their first 70% and last 30%.

The retrieval used for the published results is in data/rag/top5_<repo>.json.
Running this script writes to results/runs/rag/ instead, unless --out is given.

Usage
-----
  pip install sentence-transformers torch
  python src/rag_retrieval.py --repos pytorch keras deepspeed
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, RESULTS, ROOT, _read_jsonl, load_samples, save_json, truncate_head_tail  # noqa: E402

EMBED_MODEL = "BAAI/bge-m3"
MAX_SEQ_TOKENS = 1024
CHAR_CAP = 12000
TOP_K = 5
SELF_DUPLICATE = 0.999
CACHE = os.path.join(ROOT, "cache", "embeddings")


def embed(texts):
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(EMBED_MODEL, device=device)
    model.max_seq_length = MAX_SEQ_TOKENS
    vecs = model.encode([truncate_head_tail(t, CHAR_CAP, "\n...[TRUNCATED]...\n") for t in texts],
                        batch_size=32, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)
    return vecs.astype(np.float32)


def cached(name, texts):
    import numpy as np
    path = os.path.join(CACHE, name + ".npy")
    if os.path.exists(path):
        return np.load(path)
    os.makedirs(CACHE, exist_ok=True)
    vecs = embed(texts)
    np.save(path, vecs)
    return vecs


def main():
    import numpy as np
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repos", nargs="+", default=["pytorch", "keras", "deepspeed"])
    ap.add_argument("--out", default=os.path.join(RESULTS, "runs", "rag"))
    args = ap.parse_args()

    corpus = _read_jsonl(os.path.join(DATA, "rag", "corpus.jsonl.gz"))
    c_vecs = cached("corpus", [c["diff"] for c in corpus])
    for repo in args.repos:
        samples = load_samples(repo)
        q_vecs = cached(f"query_{repo}", [s["diff_text"] for s in samples])
        sims = q_vecs @ c_vecs.T
        rows = []
        for i, s in enumerate(samples):
            examples = []
            for j in np.argsort(-sims[i]):
                if sims[i, j] >= SELF_DUPLICATE:
                    continue
                c = corpus[j]
                examples.append({"rank": len(examples) + 1, "similarity": round(float(sims[i, j]), 4),
                                 "domain": c["domain"], "commit_id": c["commit_id"], "message": c["message"]})
                if len(examples) == TOP_K:
                    break
            rows.append({"pr_number": s["pr_number"], "examples": examples})
        path = os.path.join(args.out, f"top5_{repo}.json")
        save_json(rows, path, indent=1)
        print(f"[{repo}] {len(rows)} queries -> {path}")


if __name__ == "__main__":
    main()
