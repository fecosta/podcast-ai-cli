#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from openai import OpenAI

INDEX_FILE = Path("index.faiss")
META_FILE = Path("meta.jsonl")
CHUNKS_FILE = Path("chunks.jsonl")

# Local embedding model (your FAISS index was built with this)
LOCAL_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# OpenAI chat models
DEFAULT_CHAT_MODEL = "gpt-4o-mini"   #  [oai_citation:3‡OpenAI Platform](https://platform.openai.com/docs/models/gpt-4o-mini?utm_source=chatgpt.com)
SMART_CHAT_MODEL = "gpt-5.2"         #  [oai_citation:4‡OpenAI Platform](https://platform.openai.com/docs/models?utm_source=chatgpt.com)


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


def match_labels(meta_row: Dict[str, Any], want: List[str]) -> bool:
    if not want:
        return True
    labels = set(l.lower() for l in meta_row.get("labels", []))
    want_set = set(w.lower() for w in want)
    return len(labels.intersection(want_set)) > 0


def search(query: str, k: int, labels: List[str]) -> List[Tuple[float, int]]:
    model = SentenceTransformer(LOCAL_EMBED_MODEL)
    q = model.encode([query], normalize_embeddings=True)
    q = np.asarray(q, dtype="float32")

    index = faiss.read_index(str(INDEX_FILE))
    scores, ids = index.search(q, k * 8)  # oversample for label filtering

    meta = load_meta()
    results = []
    for score, idx in zip(scores[0].tolist(), ids[0].tolist()):
        if idx < 0 or idx >= len(meta):
            continue
        if not match_labels(meta[idx], labels):
            continue
        results.append((float(score), int(idx)))
        if len(results) >= k:
            break
    return results


def build_context_snippets(
    hits: List[Tuple[float, int]],
    meta: List[Dict[str, Any]],
    texts: List[str],
    max_chars_per_chunk: int = 1400,
) -> Tuple[str, List[str]]:
    """
    Returns:
      - context text to feed the model
      - citations list like ["episode-folder#0007", ...]
    """
    blocks = []
    citations = []
    for rank, (score, idx) in enumerate(hits, start=1):
        m = meta[idx]
        t = texts[idx]
        cite = f"{m['episode_folder']}#{m['chunk_id']}"
        citations.append(cite)

        excerpt = t[:max_chars_per_chunk]
        blocks.append(
            f"[{rank}] CITE: {cite}\n"
            f"Episode: {m.get('episode_title','').strip()}\n"
            f"Guest: {m.get('guest','').strip()}\n"
            f"Labels: {', '.join(m.get('labels', []))}\n"
            f"Text:\n{excerpt}\n"
        )
    return "\n---\n".join(blocks), citations


def answer_with_openai(question: str, context: str, citations: List[str], model: str) -> str:
    """
    Uses the Responses API. Docs:  [oai_citation:5‡OpenAI Platform](https://platform.openai.com/docs/api-reference/responses?utm_source=chatgpt.com)
    """
    client = OpenAI()

    system = (
        "You are a personal assistant answering questions about digital products "
        "using only the provided podcast transcript excerpts.\n"
        "Rules:\n"
        "- If the context is insufficient, say what’s missing and suggest a better query.\n"
        "- Do NOT invent facts.\n"
        "- When you make a claim, include citations using the format (episodes/<folder>#<chunk_id>).\n"
        "- Prefer bullet points and actionable guidance.\n"
    )

    user = (
        f"Question:\n{question}\n\n"
        f"Transcript excerpts:\n{context}\n\n"
        "Write an answer grounded in the excerpts. Include citations like "
        "(episodes/<folder>#<chunk_id>) after the sentences they support."
    )

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        # You can tune these:
        max_output_tokens=900,
    )

    # The SDK returns structured output; output_text is the easiest accessor.
    return resp.output_text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("q", nargs="+", help="your question")
    ap.add_argument("--k", type=int, default=6, help="top chunks to use")
    ap.add_argument("--label", action="append", default=[], help="filter by label (repeatable)")
    ap.add_argument("--show-text", action="store_true", help="print retrieved chunk excerpts")
    ap.add_argument("--answer", action="store_true", help="generate a synthesized answer with OpenAI")
    ap.add_argument("--smart", action="store_true", help="use a stronger (more expensive) model for answering")
    args = ap.parse_args()

    question = " ".join(args.q).strip()

    if not INDEX_FILE.exists() or not META_FILE.exists() or not CHUNKS_FILE.exists():
        print("Missing index/meta/chunks. Run 01_build_chunks.py then 02_build_index.py first.")
        return

    if args.answer and not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set.")
        return

    meta = load_meta()
    texts = load_chunks_text()

    hits = search(question, k=args.k, labels=args.label)

    print("\n=== Query ===")
    print(question)
    if args.label:
        print("Label filter:", ", ".join(args.label))

    if not hits:
        print("\nNo matches found. Try removing label filters or asking differently.")
        return

    context, cites = build_context_snippets(hits, meta, texts)

    print("\n=== Retrieved evidence ===")
    for rank, (score, idx) in enumerate(hits, start=1):
        m = meta[idx]
        cite = f"{m['episode_folder']}#{m['chunk_id']}"
        print(f"[{rank}] score={score:.3f}  cite={cite}  labels={', '.join(m.get('labels', []))}")
        if m.get("episode_title"):
            print(f"     {m['episode_title']} — {m.get('guest','')}")
        if m.get("youtube_url"):
            print(f"     {m['youtube_url']}")

    if args.show_text:
        print("\n=== Evidence excerpts ===\n")
        print(context)

    if args.answer:
        model = SMART_CHAT_MODEL if args.smart else DEFAULT_CHAT_MODEL
        print(f"\n=== Answer (model={model}) ===\n")
        ans = answer_with_openai(question, context, cites, model=model)
        print(ans)
        print("\nCitations used (available):")
        for c in cites:
            print(f"- episodes/{c}")

    else:
        print("\nTip: add --answer to synthesize an answer with citations.\n")


if __name__ == "__main__":
    main()