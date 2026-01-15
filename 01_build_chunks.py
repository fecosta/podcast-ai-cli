#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

# Multi-label taxonomy for digital product Q&A
TOPICS: Dict[str, List[str]] = {
    "Product": [
        "roadmap", "priorit", "discovery", "pmf", "positioning", "pricing", "packaging",
        "customer", "user need", "mvp", "launch", "tradeoff", "north star", "metrics",
        "retention", "activation", "onboarding", "experiment", "a/b", "ab test", "cohort",
        "jobs to be done", "jtbd",
    ],
    "Growth": [
        "activation", "retention", "acquisition", "funnel", "virality", "referral",
        "growth loop", "lifecycle", "onboarding", "conversion", "churn", "engagement",
        "ltv", "cac", "payback",
    ],
    "Marketing": [
        "marketing", "brand", "messaging", "positioning", "go-to-market", "gtm",
        "content", "seo", "category", "narrative", "launch plan", "demand gen",
    ],
    "Sales": [
        "sales", "pipeline", "deal", "enterprise", "negotiation", "objection",
        "account", "quota", "procurement", "contract", "pricing discussion",
    ],
    "Data": [
        "analytics", "dashboard", "instrument", "event", "tracking", "metric",
        "kpi", "experiment", "a/b", "ab test", "causal", "inference", "sample size",
        "statistical", "cohort", "segmentation",
    ],
    "Tech": [
        "architecture", "api", "backend", "frontend", "infrastructure", "cloud",
        "kubernetes", "docker", "latency", "scalability", "incident", "on-call",
        "technical debt", "refactor", "reliability", "security",
        "machine learning", "ml", "ai", "llm", "prompt", "embedding", "vector",
    ],
    "Design": [
        "ux", "ui", "usability", "prototype", "wireframe", "design system",
        "accessibility", "interaction", "research", "user testing", "information architecture",
    ],
    "Leadership": [
        "leadership", "manage", "manager", "hiring", "interview", "performance",
        "feedback", "culture", "alignment", "stakeholder", "decision", "conflict",
        "org design", "team", "one-on-one", "1:1", "strategy offsite",
    ],
    "Finance": [
        "revenue", "profit", "margin", "unit economics", "burn", "runway",
        "funding", "valuation", "cap table", "budget", "forecast", "cash flow",
        "pricing", "gross margin",
    ],
    "Operations": [
        "process", "execution", "planning", "cadence", "okr", "okrs",
        "project management", "program", "cross-functional", "handoff", "tooling",
    ],
    "Career": [
        "career", "leveling", "promotion", "compensation", "resume", "portfolio",
        "mentorship", "manager track", "ic track", "interview process",
    ],
}

DEFAULT_MIN_SCORE = 0.22  # tune: lower -> more labels per chunk
CHUNK_WORDS = 650
CHUNK_OVERLAP = 130

OUT_FILE = Path("chunks.jsonl")


@dataclass
class Episode:
    folder: str
    path: Path
    guest: str
    title: str
    youtube_url: str
    text: str


def parse_frontmatter(md: str) -> Tuple[dict, str]:
    if not md.startswith("---"):
        return {}, md
    parts = md.split("---", 2)
    if len(parts) < 3:
        return {}, md
    fm_text, body = parts[1], parts[2]
    try:
        return (yaml.safe_load(fm_text) or {}), body
    except Exception:
        return {}, md


def load_episodes(episodes_dir: Path) -> List[Episode]:
    eps: List[Episode] = []
    for p in episodes_dir.glob("*/transcript.md"):
        md = p.read_text(encoding="utf-8", errors="replace")
        fm, body = parse_frontmatter(md)

        guest = str(fm.get("guest", p.parent.name)).strip()
        title = str(fm.get("title", "")).strip()
        youtube_url = str(fm.get("youtube_url", "")).strip()

        # light cleanup
        text = re.sub(r"\s+", " ", body).strip()
        eps.append(Episode(p.parent.name, p, guest, title, youtube_url, text))
    return eps


def chunk_text(text: str, chunk_words: int, overlap: int) -> List[str]:
    words = text.split()
    if not words:
        return []
    step = max(1, chunk_words - overlap)
    chunks = []
    for start in range(0, len(words), step):
        end = min(len(words), start + chunk_words)
        c = " ".join(words[start:end]).strip()
        if c:
            chunks.append(c)
        if end == len(words):
            break
    return chunks


def score_topics(chunk: str) -> Dict[str, float]:
    t = chunk.lower()
    scores: Dict[str, float] = {}
    for topic, kws in TOPICS.items():
        hits = 0
        for kw in kws:
            if kw in t:
                hits += 1
        # saturating score 0..1
        score = 1 - (2.71828 ** (-hits / 4.0))
        scores[topic] = float(score)
    return scores


def pick_labels(scores: Dict[str, float], min_score: float) -> List[str]:
    labels = [k for k, v in scores.items() if v >= min_score]
    labels.sort(key=lambda k: scores[k], reverse=True)
    return labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="sources/lennys", help="Path to transcripts repo root")
    ap.add_argument("--source-id", default="lennys", help="Short id used in metadata/citations")
    ap.add_argument("--out", default=str(OUT_FILE), help="Output jsonl file")
    args = ap.parse_args()

    source_root = Path(args.source)
    episodes_dir = source_root / "episodes"

    if not episodes_dir.exists():
        print(f"Could not find episodes/ at: {episodes_dir}")
        return

    episodes = load_episodes(episodes_dir)
    if not episodes:
        print(f"No episodes found under: {episodes_dir}/*/transcript.md")
        return

    out_path = Path(args.out)

    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for ep in episodes:
            chunks = chunk_text(ep.text, CHUNK_WORDS, CHUNK_OVERLAP)
            for i, ch in enumerate(chunks):
                scores = score_topics(ch)
                labels = pick_labels(scores, DEFAULT_MIN_SCORE) or ["General"]

                rec = {
                    "source_id": args.source_id,
                    "source_root": str(source_root),
                    "episode_folder": ep.folder,
                    "transcript_path": str(ep.path.relative_to(source_root)),
                    "episode_title": ep.title,
                    "guest": ep.guest,
                    "youtube_url": ep.youtube_url,
                    "chunk_id": f"{i:04d}",
                    "text": ch,
                    "labels": labels,
                    "scores": scores,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1

    print(f"Wrote {out_path} with {n} chunks.")
    print(f"Source: {source_root}")


if __name__ == "__main__":
    main()