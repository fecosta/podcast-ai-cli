#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = Path("chunks.jsonl")
INDEX_FILE = Path("index.faiss")
META_FILE = Path("meta.jsonl")

# Good default; change if you want a larger/better model later
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

BATCH = 64


def load_chunks() -> List[Dict[str, Any]]:
    out = []
    with CHUNKS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main() -> None:
    if not CHUNKS_FILE.exists():
        print("Missing chunks.jsonl. Run 01_build_chunks.py first.")
        return

    chunks = load_chunks()
    texts = [c["text"] for c in chunks]

    model = SentenceTransformer(EMBED_MODEL)
    dim = model.get_sentence_embedding_dimension()

    # Cosine similarity => normalize vectors then use inner product
    index = faiss.IndexFlatIP(dim)

    # Write meta in the exact same order as vectors
    with META_FILE.open("w", encoding="utf-8") as mf:
        for c in chunks:
            mf.write(json.dumps({
                "episode_folder": c["episode_folder"],
                "transcript_path": c["transcript_path"],
                "episode_title": c["episode_title"],
                "guest": c["guest"],
                "youtube_url": c.get("youtube_url", ""),
                "chunk_id": c["chunk_id"],
                "labels": c["labels"],
            }, ensure_ascii=False) + "\n")

    # Embed in batches
    all_vecs = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i+BATCH]
        vecs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_vecs.append(np.asarray(vecs, dtype="float32"))

    V = np.vstack(all_vecs)
    index.add(V)

    faiss.write_index(index, str(INDEX_FILE))
    print(f"Wrote {INDEX_FILE} with {index.ntotal} vectors and {META_FILE} metadata rows.")


if __name__ == "__main__":
    main()