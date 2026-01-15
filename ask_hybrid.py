#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import faiss
from openai import OpenAI

# Files you already have (OpenAI embedding FAISS index)
FAISS_INDEX = Path("index_openai.faiss")
META_FILE = Path("meta_openai.jsonl")
CHUNKS_FILE = Path("chunks.jsonl")

# New keyword index
FTS_DB = Path("chunks_fts.sqlite")

# Models
EMBEDDINGS_MODEL = "text-embedding-3-small"
DEFAULT_CHAT_MODEL = "gpt-4o-mini"
SMART_CHAT_MODEL = "gpt-5.2"

# Your taxonomy (must match what you used when building chunks.jsonl)
TAXONOMY = [
    "Product", "Growth", "Marketing", "Sales", "Data", "Tech",
    "Design", "Leadership", "Finance", "Operations", "Career", "General"
]


def l2_normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms


def load_meta() -> List[Dict[str, Any]]:
    meta = []
    with META_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))
    return meta


def load_chunks_text() -> List[str]:
    texts = []
    with CHUNKS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])
    return texts


def match_labels(labels_in_row: List[str], want: List[str]) -> bool:
    if not want:
        return True
    s = set(l.lower() for l in labels_in_row)
    w = set(x.lower() for x in want)
    return len(s.intersection(w)) > 0


# ---------- Keyword search (SQLite FTS5) ----------

def keyword_search(query: str, top_n: int) -> List[Tuple[int, float]]:
    """
    Returns list of (rowid, kw_score). Higher is better.
    We use bm25() from FTS5; lower bm25 means better match, so we invert it.
    """
    if not FTS_DB.exists():
        return []

    # Sanitize query: FTS5 MATCH can fail on special characters like "?" or "-"
    # We'll strip common punctuation and keep it simple.
    import re
    clean_query = re.sub(r'[^\w\s]', ' ', query).strip()
    if not clean_query:
        return []

    con = sqlite3.connect(str(FTS_DB))
    con.row_factory = sqlite3.Row

    # bm25() returns smaller for better -> convert to score with 1/(1+bm25)
    rows = con.execute(
        """
        SELECT c.rowid as rowid, bm25(chunks_fts) AS bm
        FROM chunks_fts
        JOIN chunks c ON c.rowid = chunks_fts.rowid
        WHERE chunks_fts MATCH ?
        ORDER BY bm ASC
        LIMIT ?
        """,
        (clean_query, top_n),
    ).fetchall()

    con.close()

    out = []
    for r in rows:
        bm = float(r["bm"])
        score = 1.0 / (1.0 + max(bm, 0.0))
        out.append((int(r["rowid"]) - 1, score))  # rowid starts at 1; our meta/text arrays are 0-based
    return out


# ---------- Semantic search (FAISS with OpenAI embeddings) ----------

def embed_query(client: OpenAI, query: str) -> np.ndarray:
    resp = client.embeddings.create(
        model=EMBEDDINGS_MODEL,
        input=query,
        encoding_format="float",
    )
    v = np.array(resp.data[0].embedding, dtype=np.float32).reshape(1, -1)
    return l2_normalize(v)


def semantic_search(client: OpenAI, query: str, top_n: int) -> List[Tuple[int, float]]:
    q = embed_query(client, query)
    index = faiss.read_index(str(FAISS_INDEX))
    scores, ids = index.search(q, top_n)
    out = []
    for s, idx in zip(scores[0].tolist(), ids[0].tolist()):
        if idx >= 0:
            out.append((int(idx), float(s)))  # cosine via inner product
    return out


# ---------- Merge + filter ----------

def merge_scores(
    sem: List[Tuple[int, float]],
    kw: List[Tuple[int, float]],
    alpha: float = 0.75,   # weight semantic
    beta: float = 0.25,    # weight keyword
) -> Dict[int, float]:
    scores: Dict[int, float] = {}
    for idx, s in sem:
        scores[idx] = scores.get(idx, 0.0) + alpha * s
    for idx, s in kw:
        scores[idx] = scores.get(idx, 0.0) + beta * s
    return scores


def shortlist(
    merged: Dict[int, float],
    meta: List[Dict[str, Any]],
    want_labels: List[str],
    top_m: int,
) -> List[Tuple[int, float]]:
    items = sorted(merged.items(), key=lambda x: x[1], reverse=True)
    out: List[Tuple[int, float]] = []
    for idx, score in items:
        if idx < 0 or idx >= len(meta):
            continue
        if not match_labels(meta[idx].get("labels", []), want_labels):
            continue
        out.append((idx, float(score)))
        if len(out) >= top_m:
            break
    return out


# ---------- Auto-label ----------

def auto_labels(client: OpenAI, question: str) -> List[str]:
    prompt = (
        "Choose the most relevant labels for the question. "
        "Return JSON only: {\"labels\": [..]}.\n\n"
        f"Available labels: {TAXONOMY}\n\n"
        f"Question: {question}"
    )
    resp = client.responses.create(
        model=DEFAULT_CHAT_MODEL,
        input=prompt,
        max_output_tokens=120,
    )
    text = resp.output_text.strip()
    try:
        obj = json.loads(text)
        labels = obj.get("labels", [])
        labels = [l for l in labels if l in TAXONOMY and l != "General"]
        return labels[:4]  # keep it tight
    except Exception:
        return []


# ---------- Reranker ----------

def rerank_with_llm(
    client: OpenAI,
    question: str,
    candidates: List[Tuple[int, float]],
    meta: List[Dict[str, Any]],
    texts: List[str],
    model: str,
    max_chars_each: int = 900,
) -> List[int]:
    """
    Returns reordered list of indices.
    """
    items = []
    for i, (idx, score) in enumerate(candidates, start=1):
        m = meta[idx]
        cite = f"episodes/{m['episode_folder']}#{m['chunk_id']}"
        snippet = texts[idx][:max_chars_each]
        items.append(
            f"Item {i}\nCITE: {cite}\nTitle: {m.get('episode_title','')}\nGuest: {m.get('guest','')}\nText: {snippet}"
        )

    prompt = (
        "You are a reranker for retrieval-augmented QA.\n"
        "Task: reorder the items by how useful they are to answer the question.\n"
        "Return JSON only: {\"order\": [item_numbers...]}\n\n"
        f"Question: {question}\n\n"
        "Items:\n" + "\n\n---\n\n".join(items)
    )

    resp = client.responses.create(
        model=model,
        input=prompt,
        max_output_tokens=220,
    )
    out = resp.output_text.strip()
    try:
        obj = json.loads(out)
        order = obj.get("order", [])
        # map item numbers -> idx
        reordered = []
        for n in order:
            if isinstance(n, int) and 1 <= n <= len(candidates):
                reordered.append(candidates[n-1][0])
        # fallback: append anything missing
        seen = set(reordered)
        for idx, _ in candidates:
            if idx not in seen:
                reordered.append(idx)
        return reordered
    except Exception:
        # no rerank
        return [idx for idx, _ in candidates]


# ---------- Answering ----------

def build_context(indices: List[int], meta: List[Dict[str, Any]], texts: List[str], max_chars: int = 1400) -> str:
    blocks = []
    for rank, idx in enumerate(indices, start=1):
        m = meta[idx]
        cite = f"episodes/{m['episode_folder']}#{m['chunk_id']}"
        excerpt = texts[idx][:max_chars]
        blocks.append(
            f"[{rank}] CITE: {cite}\n"
            f"Episode: {m.get('episode_title','').strip()}\n"
            f"Guest: {m.get('guest','').strip()}\n"
            f"Labels: {', '.join(m.get('labels', []))}\n"
            f"Text:\n{excerpt}\n"
        )
    return "\n---\n".join(blocks)


def answer(client: OpenAI, question: str, context: str, model: str) -> str:
    system = (
        "You are a personal assistant for digital product questions.\n"
        "Use ONLY the provided excerpts.\n"
        "Rules:\n"
        "- Do not invent facts.\n"
        "- If insufficient evidence, say what’s missing and suggest a better query.\n"
        "- Add citations after the sentences they support, using the provided CITE values.\n"
        "- Prefer structured, actionable output.\n"
    )
    user = (
        f"Question:\n{question}\n\n"
        f"Excerpts:\n{context}\n\n"
        "Answer with citations like (episodes/<folder>#<chunk_id>)."
    )
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_output_tokens=950,
    )
    return resp.output_text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("q", nargs="+", help="your question")
    ap.add_argument("--k", type=int, default=6, help="chunks to answer with")
    ap.add_argument("--label", action="append", default=[], help="label filter (repeatable)")
    ap.add_argument("--auto-label", action="store_true", help="let the model pick labels for the question")
    ap.add_argument("--semantic-n", type=int, default=30, help="semantic candidates")
    ap.add_argument("--keyword-n", type=int, default=30, help="keyword candidates")
    ap.add_argument("--merge-alpha", type=float, default=0.75, help="semantic weight")
    ap.add_argument("--merge-beta", type=float, default=0.25, help="keyword weight")
    ap.add_argument("--rerank", action="store_true", help="rerank shortlist with OpenAI")
    ap.add_argument("--smart", action="store_true", help="use stronger model for rerank/answer")
    ap.add_argument("--show-evidence", action="store_true", help="print evidence excerpts")
    ap.add_argument("--no-answer", action="store_true", help="only retrieve, don't synthesize")
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set.")
        return

    # Ensure files exist
    needed = [FAISS_INDEX, META_FILE, CHUNKS_FILE, FTS_DB]
    missing = [p for p in needed if not p.exists()]
    if missing:
        print("Missing required files:")
        for p in missing:
            print("-", p)
        print("\nRun:")
        print("  python 01_build_chunks.py")
        print("  python 02_build_index_openai.py")
        print("  python 03_build_fts.py")
        return

    client = OpenAI()
    meta = load_meta()
    texts = load_chunks_text()

    question = " ".join(args.q).strip()

    want_labels = list(args.label)
    if args.auto_label:
        picked = auto_labels(client, question)
        # If user also provided labels, we keep both (union)
        want_labels = sorted(set(want_labels + picked))
        if want_labels:
            print("Auto labels:", ", ".join(want_labels))

    sem = semantic_search(client, question, top_n=args.semantic_n)
    kw = keyword_search(question, top_n=args.keyword_n)

    merged = merge_scores(sem, kw, alpha=args.merge_alpha, beta=args.merge_beta)
    short = shortlist(merged, meta, want_labels, top_m=max(args.k * 4, 16))

    if not short:
        print("No matches found. Try removing label filters or rephrasing.")
        return

    # Candidate indices (pre-rerank)
    cand_indices = [idx for idx, _ in short]

    model = SMART_CHAT_MODEL if args.smart else DEFAULT_CHAT_MODEL

    if args.rerank:
        reranked = rerank_with_llm(client, question, short, meta, texts, model=model)
        final_indices = reranked[: args.k]
    else:
        final_indices = cand_indices[: args.k]

    print("\n=== Retrieved evidence ===")
    for rank, idx in enumerate(final_indices, start=1):
        m = meta[idx]
        cite = f"episodes/{m['episode_folder']}#{m['chunk_id']}"
        print(f"[{rank}] cite={cite} labels={', '.join(m.get('labels', []))}")
        if m.get("episode_title"):
            print(f"     {m['episode_title']} — {m.get('guest','')}")
        if m.get("youtube_url"):
            print(f"     {m['youtube_url']}")

    context = build_context(final_indices, meta, texts)

    if args.show_evidence:
        print("\n=== Evidence excerpts ===\n")
        print(context)

    if args.no_answer:
        return

    print(f"\n=== Answer (model={model}) ===\n")
    print(answer(client, question, context, model=model))


if __name__ == "__main__":
    main()