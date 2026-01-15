# Podcast AI CLI

![GitHub stars](https://img.shields.io/github/stars/fecosta/podcast-ai-cli)
![GitHub issues](https://img.shields.io/github/issues/fecosta/podcast-ai-cli)

A **local, CLI-based Retrieval-Augmented Generation (RAG) assistant** for answering high-quality questions about **product, growth, tech, leadership, design, finance**, and digital product work — grounded in long-form podcast transcripts (starting with *Lenny’s Podcast*).

This repository contains **only the AI/CLI tooling**. Transcript content lives in a separate repository and is consumed as a **Git submodule**.

---

## 🧠 Quick Start

```bash
git clone https://github.com/fecosta/podcast-ai-cli.git
cd podcast-ai-cli
git submodule update --init --remote

pip install openai numpy faiss-cpu
export OPENAI_API_KEY="YOUR_KEY"

python 01_build_chunks.py --source sources/lennys --source-id lennys
python 02_build_index_openai.py
python 03_build_fts.py

python ask_hybrid.py "How do I reduce churn without hurting activation?" --auto-label --rerank
```

---

## What this project does

- Ingests podcast transcripts from an external repo
- Chunks and labels content by topic
- Builds:
  - a **semantic vector index** (OpenAI embeddings + FAISS)
  - a **keyword index** (SQLite FTS5)
- Answers questions via a **hybrid retrieval pipeline**:
  - semantic search + keyword search
  - optional LLM reranking
  - grounded answers with explicit citations

Everything runs **locally from the CLI**.

---

## Repository structure

```
podcast-ai-cli/
├── ask_hybrid.py                 # Main CLI: ask questions
├── 01_build_chunks.py            # Ingest + chunk + label transcripts
├── 02_build_index_openai.py      # Build FAISS index + metadata
├── 03_build_fts.py               # Build SQLite FTS keyword index
├── sources/
│   └── lennys/                   # Git submodule (transcripts repo)
│       └── episodes/**/transcript.md
└── README.md
```

Generated files (usually gitignored):
- `chunks.jsonl`
- `meta_openai.jsonl`
- `index_openai.faiss`
- `chunks_fts.sqlite`

---

## Transcript source (submodule)

Transcript data is maintained in a separate repository:
👉 https://github.com/ChatPRD/lennys-podcast-transcripts

It is added here as a Git submodule:

```bash
git submodule add https://github.com/ChatPRD/lennys-podcast-transcripts.git sources/lennys
```

Expected structure:

```
sources/lennys/
└── episodes/
    └── <episode-folder>/
        └── transcript.md
```

---

## Build pipeline

Run these steps **in order** whenever transcripts change.

### 1) Chunk and label transcripts

```bash
python 01_build_chunks.py --source sources/lennys --source-id lennys
```

Outputs: `chunks.jsonl`

Each chunk includes text, topic labels, episode metadata, and source identifiers.

---

### 2) Build semantic index (FAISS)

```bash
python 02_build_index_openai.py
```

Outputs:
- `index_openai.faiss`
- `meta_openai.jsonl`

Uses OpenAI embeddings (`text-embedding-3-small`).

---

### 3) Build keyword index (SQLite FTS5)

```bash
python 03_build_fts.py
```

Output: `chunks_fts.sqlite`

Enables exact-term and name-based retrieval.

---

## Asking questions

### Recommended usage

```bash
python ask_hybrid.py "How do I reduce churn without hurting activation?" --auto-label --rerank
```

### Smart mode (higher accuracy, higher cost)

```bash
python ask_hybrid.py "How do strong PMs decide what NOT to build?" --auto-label --rerank --smart
```

---

## CLI flags

| Flag | Description |
|-----|-------------|
| `--auto-label` | Let the LLM choose topic filters |
| `--label <X>` | Manually filter by label (repeatable) |
| `--rerank` | Rerank retrieved chunks with LLM |
| `--smart` | Use stronger model for rerank + answer |
| `--k` | Number of chunks used to answer (default 6) |
| `--show-evidence` | Print retrieved excerpts |
| `--no-answer` | Retrieval only (no synthesis) |

---

## How retrieval works

1. Semantic search (FAISS + OpenAI embeddings)
2. Keyword search (SQLite FTS5)
3. Score merge (weighted semantic + keyword)
4. Label filtering (manual or auto)
5. Optional reranking (LLM)
6. Answer generation using **only retrieved excerpts**

---

## Citations format

All answers are grounded with explicit citations:

```
lennys:episodes/<episode-folder>#<chunk_id>
```

If `source_id` is missing, it falls back to:

```
episodes/<episode-folder>#<chunk_id>
```

The assistant is instructed **not to invent facts** and to cite every claim.

---

## Topic taxonomy

Default labels:

- Product
- Growth
- Marketing
- Sales
- Data
- Tech
- Design
- Leadership
- Finance
- Operations
- Career
- General

You can extend or refine this taxonomy in `01_build_chunks.py`.

---

## Adding new sources

You can ingest additional corpora by:

1. Adding another submodule:

```
sources/my-podcast/
└── episodes/**/transcript.md
```

2. Running ingest with a new source id:

```bash
python 01_build_chunks.py --source sources/my-podcast --source-id mypodcast
```

All downstream steps remain the same.

---

## Design goals

- Local-first
- Transparent retrieval
- Explicit citations
- Easy to extend to new corpora
- No hidden magic

This is intended to be a **thinking tool**, not a chat toy.

---

## Roadmap ideas

- Incremental ingest (embed only changed chunks)
- Speaker-aware chunking
- Answer verification pass
- Evaluation harness
- Packaging as `podcast-ai` CLI

---

## License

MIT © Felipe Costa

