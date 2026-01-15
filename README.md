# Podcast AI CLI

A local, CLI-based Retrieval-Augmented Generation (RAG) assistant for answering questions about **product, growth, tech, leadership, design, finance**, and more — grounded in long-form podcast transcripts (starting with Lenny’s Podcast).

This repository contains **only the AI/CLI tooling**. Transcript content lives in a separate repository and is consumed as a **Git submodule**.

---

## What this project does

- Ingests podcast transcripts from an external repo
- Chunks and labels the content by topic
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
├── chunks.jsonl                  # Chunked dataset (generated)
├── meta_openai.jsonl             # Metadata aligned with FAISS index
├── index_openai.faiss            # Vector index
├── chunks_fts.sqlite             # Keyword index
├── sources/
│   └── lennys/                   # Git submodule (transcripts repo)
│       └── episodes/**/transcript.md
└── README.md
```

Generated files (`*.faiss`, `*.sqlite`, `*.jsonl`) are usually **gitignored**.

---

## Transcript source (submodule)

This repo expects transcripts to live in a separate repository and be added as a submodule.

Default setup:

```
sources/lennys/
└── episodes/
    └── <episode-folder>/
        └── transcript.md
```

Add/update the submodule:

```bash
git submodule add https://github.com/ChatPRD/lennys-podcast-transcripts.git sources/lennys
# later
git submodule update --remote
```

---

## Installation

### Requirements

- Python 3.10+
- OpenAI API key

### Install dependencies

```bash
pip install openai numpy faiss-cpu
```

Set your API key:

```bash
export OPENAI_API_KEY="your_key_here"
```

---

## Ingest & build pipeline

Run these steps **in order** whenever transcripts change.

### 1) Chunk and label transcripts

```bash
python 01_build_chunks.py --source sources/lennys --source-id lennys
```

Outputs:
- `chunks.jsonl`

Each chunk includes:
- text
- topic labels (Product, Growth, etc.)
- episode + guest metadata
- source identifier

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

Output:
- `chunks_fts.sqlite`

This enables exact-term and name-based retrieval.

---

## Asking questions

### Basic usage

```bash
python ask_hybrid.py "How do I reduce churn without hurting activation?"
```

### With auto-labeling (recommended)

```bash
python ask_hybrid.py "How do I reduce churn without hurting activation?" --auto-label
```

### With reranking (higher quality)

```bash
python ask_hybrid.py "How do I reduce churn without hurting activation?" --auto-label --rerank
```

### Smart mode (stronger model, higher cost)

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

## Retrieval pipeline (how answers are built)

1. **Semantic search** (FAISS + OpenAI embeddings)
2. **Keyword search** (SQLite FTS5)
3. **Score merge** (weighted semantic + keyword)
4. **Label filtering** (manual or auto)
5. **Optional reranking** (LLM)
6. **Answer generation** using only retrieved excerpts

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

Default high-level labels:

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

You can add more transcript or document sources by:

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