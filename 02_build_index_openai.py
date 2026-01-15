#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import faiss
from openai import OpenAI

CHUNKS_FILE = Path("chunks.jsonl")
INDEX_FILE = Path("index_openai.faiss")
META_FILE = Path("meta_openai.jsonl")

EMBEDDINGS_MODEL = "text-embedding-3-small"  #  [oai_citation:3‡OpenAI Platform](https://platform.openai.com/docs/models/text-embedding-3-small?utm_source=chatgpt.com)
BATCH = 96  # keep reasonable to avoid request size issues


def l2_normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms


def load_chunks() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with CHUNKS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set.")
        return

    if not CHUNKS_FILE.exists():
        print("Missing chunks.jsonl. Run 01_build_chunks.py first.")
        return

    client = OpenAI()

    chunks = load_chunks()
    texts = [c["text"] for c in chunks]

    # Write metadata in the same order as embeddings
    with META_FILE.open("w", encoding="utf-8") as mf:
        for c in chunks:
            mf.write(
                json.dumps(
                    {
                        "source_id": c.get("source_id"),
                        "source_root": c.get("source_root"),
                        "episode_folder": c["episode_folder"],
                        "transcript_path": c["transcript_path"],
                        "episode_title": c["episode_title"],
                        "guest": c["guest"],
                        "youtube_url": c.get("youtube_url", ""),
                        "chunk_id": c["chunk_id"],
                        "labels": c["labels"],
                        },
                    ensure_ascii=False,
                )
                + "\n"
            )

    all_vecs: List[np.ndarray] = []

    for i in range(0, len(texts), BATCH):
        batch = texts[i : i + BATCH]
        resp = client.embeddings.create(
            model=EMBEDDINGS_MODEL,
            input=batch,
            encoding_format="float",
        )
        # resp.data[j].embedding corresponds to batch[j]
        vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
        vecs = l2_normalize(vecs)  # cosine via inner product
        all_vecs.append(vecs)

    V = np.vstack(all_vecs)
    dim = V.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(V)

    faiss.write_index(index, str(INDEX_FILE))
    print(f"Wrote {INDEX_FILE} with {index.ntotal} vectors (dim={dim}) and {META_FILE} metadata rows.")


if __name__ == "__main__":
    main()