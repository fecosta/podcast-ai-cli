#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import faiss
from openai import OpenAI

INDEX_FILE = Path("index_openai.faiss")
META_FILE = Path("meta_openai.jsonl")
CHUNKS_FILE = Path("chunks.jsonl")

EMBEDDINGS_MODEL = "text-embedding-3-small"  #  [oai_citation:5‡OpenAI Platform](https://platform.openai.com/docs/models/text-embedding-3-small?utm_source=chatgpt.com)
DEFAULT_CHAT_MODEL = "gpt-4o-mini"          #  [oai_citation:6‡OpenAI Platform](https://platform.openai.com/docs/models/gpt-4o-mini?utm_source=chatgpt.com)
SMART_CHAT_MODEL = "gpt-5.2"                #  [oai_citation:7‡OpenAI Platform](https://platform.openai.com/docs/models?utm_source=chatgpt.com)


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


def match_labels(meta_row: Dict[str, Any], want: List[str]) -> bool:
    if not want:
        return True
    labels = set(l.lower() for l in meta_row.get("labels", []))
    want_set = set(w.lower() for w in want)
    return len(labels.intersection(want_set)) > 0


def embed_query(client: OpenAI, query: str) -> np.ndarray:
    resp = client.embeddings.create(
        model=EMBEDDINGS_MODEL,
        input=query,
        encoding_format="float",
    )
    v = np.array(resp.data[0].embedding, dtype=np.float32).reshape(1, -1)
    return l2_normalize(v)


def search(client: OpenAI, query: str, k: int, labels: List[str]) -> List[Tuple[float, int]]:
    q = embed_query(client, query)

    index = faiss.read_index(str(INDEX_FILE))
    scores, ids = index.search(q, k * 8)  # oversample so label filtering still returns k

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


def build_context(hits: List[Tuple[float, int]], meta: List[Dict[str, Any]], texts: List[str], max_chars: int = 1400) -> str:
    blocks = []
    for rank, (score, idx) in enumerate(hits, start=1):
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


def answer_with_openai(client: OpenAI, question: str, context: str, model: str) -> str:
    # Uses Responses API  [oai_citation:8‡OpenAI Platform](https://platform.openai.com/docs/api-reference/responses?utm_source=chatgpt.com)
    system = (
        "You are a personal assistant answering questions about digital products using ONLY the provided excerpts.\n"
        "Rules:\n"
        "- Do not invent facts.\n"
        "- If the excerpts are insufficient, say what’s missing.\n"
        "- Add citations after the sentences they support, using the provided CITE values.\n"
    )

    user = (
        f"Question:\n{question}\n\n"
        f"Transcript excerpts:\n{context}\n\n"
        "Answer clearly and practically. Use citations like (episodes/<folder>#<chunk_id>)."
    )

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_output_tokens=900,
    )
    return resp.output_text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("q", nargs="+", help="your question")
    ap.add_argument("--k", type=int, default=6, help="top chunks to use")
    ap.add_argument("--label", action="append", default=[], help="filter by label (repeatable)")
    ap.add_argument("--show-text", action="store_true", help="print evidence excerpts")
    ap.add_argument("--answer", action="store_true", help="generate an answer with OpenAI")
    ap.add_argument("--smart", action="store_true", help="use a stronger model for answering")
    args = ap.parse_args()

    question = " ".join(args.q).strip()

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set.")
        return

    if not (INDEX_FILE.exists() and META_FILE.exists() and CHUNKS_FILE.exists()):
        print("Missing index_openai.faiss / meta_openai.jsonl / chunks.jsonl.")
        print("Run: python 01_build_chunks.py && python 02_build_index_openai.py")
        return

    client = OpenAI()
    meta = load_meta()
    texts = load_chunks_text()

    hits = search(client, question, k=args.k, labels=args.label)

    print("\n=== Query ===")
    print(question)
    if args.label:
        print("Label filter:", ", ".join(args.label))

    if not hits:
        print("\nNo matches found. Try removing label filters or rephrasing.")
        return

    print("\n=== Retrieved evidence ===")
    for rank, (score, idx) in enumerate(hits, start=1):
        m = meta[idx]
        cite = f"episodes/{m['episode_folder']}#{m['chunk_id']}"
        print(f"[{rank}] score={score:.3f}  cite={cite}  labels={', '.join(m.get('labels', []))}")
        if m.get("episode_title"):
            print(f"     {m['episode_title']} — {m.get('guest','')}")
        if m.get("youtube_url"):
            print(f"     {m['youtube_url']}")

    context = build_context(hits, meta, texts)

    if args.show_text:
        print("\n=== Evidence excerpts ===\n")
        print(context)

    if args.answer:
        model = SMART_CHAT_MODEL if args.smart else DEFAULT_CHAT_MODEL
        print(f"\n=== Answer (model={model}) ===\n")
        print(answer_with_openai(client, question, context, model=model))


if __name__ == "__main__":
    main()