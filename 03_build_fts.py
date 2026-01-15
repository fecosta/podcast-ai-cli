#!/usr/bin/env python3
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

CHUNKS_FILE = Path("chunks.jsonl")
DB_FILE = Path("chunks_fts.sqlite")


def main() -> None:
    if not CHUNKS_FILE.exists():
        print("Missing chunks.jsonl. Run 01_build_chunks.py first.")
        return

    if DB_FILE.exists():
        DB_FILE.unlink()

    con = sqlite3.connect(str(DB_FILE))
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")

    # Base table (now includes source_id + transcript_path)
    con.execute("""
    CREATE TABLE chunks (
        rowid INTEGER PRIMARY KEY,
        source_id TEXT,
        transcript_path TEXT,
        episode_folder TEXT,
        chunk_id TEXT,
        episode_title TEXT,
        guest TEXT,
        youtube_url TEXT,
        labels_json TEXT,
        text TEXT
    )
    """)

    # FTS5 virtual table (keyword search)
    # Indexing a few metadata fields improves recall for names/episodes.
    con.execute("""
    CREATE VIRTUAL TABLE chunks_fts USING fts5(
        text,
        episode_title,
        guest,
        episode_folder,
        source_id,
        content='chunks',
        content_rowid='rowid'
    )
    """)

    rows = 0
    with CHUNKS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)

            con.execute(
                """
                INSERT INTO chunks (
                    source_id, transcript_path, episode_folder, chunk_id,
                    episode_title, guest, youtube_url, labels_json, text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    obj.get("source_id", "lennys"),
                    obj.get("transcript_path", ""),
                    obj.get("episode_folder", ""),
                    obj.get("chunk_id", ""),
                    obj.get("episode_title", ""),
                    obj.get("guest", ""),
                    obj.get("youtube_url", ""),
                    json.dumps(obj.get("labels", []), ensure_ascii=False),
                    obj.get("text", ""),
                ),
            )
            rows += 1

    # Build FTS index from base table content
    con.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
    con.commit()
    con.close()

    print(f"Wrote {DB_FILE} with {rows} chunks indexed for keyword search.")


if __name__ == "__main__":
    main()