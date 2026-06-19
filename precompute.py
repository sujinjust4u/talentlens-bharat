#!/usr/bin/env python3
"""
TalentLens Bharat — precompute.py
==================================
Offline embedding pre-computation script.

Run this ONCE before submission to generate:
  - precomputed/embeddings.npy      (100K × 384 float16, ~73 MB)
  - precomputed/jd_embedding.npy    (384, float32)
  - precomputed/candidate_ids.json  (ordered list of candidate_ids)

Usage:
  python precompute.py
  python precompute.py --candidates ./candidates.jsonl.gz --batch-size 256
  python precompute.py --candidates ./candidates.jsonl --jd data/parsed_jd.json

This script DOES use sentence-transformers (network for model download on first run).
The output files are used by rank.py which has ZERO network dependencies.

Expected runtime: ~10-20 minutes for 100K candidates on M2 MacBook.
Expected memory: ~2 GB peak (model + batch embeddings).
"""

import argparse
import gzip
import json
import os
import sys
import time

import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, fast, good quality


# ===========================================================================
# File Opener Helper (supports .jsonl and .jsonl.gz)
# ===========================================================================
def _open_candidates_file(path: str):
    """Returns an appropriate file handle for .jsonl or .jsonl.gz files."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    else:
        return open(path, "r", encoding="utf-8")


# ===========================================================================
# Profile Text Construction (Real Schema)
# ===========================================================================
def build_jd_profile_text(jd: dict) -> str:
    """
    Transforms the parsed JD JSON into a coherent textual document
    for embedding. Same logic as stage2_ranker.py but standalone.
    """
    parts = []

    seniority = jd.get("seniority", "Mid-Level")
    domain = jd.get("domain", "General Technology")
    parts.append(f"Looking for a {seniority} level professional working in the {domain} field.")

    req_skills = jd.get("required_skills", [])
    if req_skills:
        parts.append(f"Required core skills and technologies: {', '.join(req_skills)}.")

    nice_to_have = jd.get("nice_to_have", [])
    if nice_to_have:
        parts.append(f"Nice to have preferences: {', '.join(nice_to_have)}.")

    implicit = jd.get("implicit_signals", [])
    if implicit:
        parts.append(f"Contextual requirements: {', '.join(implicit)}.")

    return " ".join(parts)


def build_candidate_profile_text(raw: dict) -> str:
    """
    Transforms a candidate's nested JSON record into a coherent textual
    document suitable for sentence-transformer embedding.

    Uses the REAL schema fields:
      - profile.current_title, profile.current_industry
      - profile.years_of_experience
      - skills[].name
      - career_history[].title (non-current)
      - profile.headline
      
    NOTE: We intentionally SKIP the long summary field.
    Summaries average 530 chars and cause 10-15x slower encoding.
    The headline + skills + title capture the same semantic signal
    in a fraction of the tokens.
    """
    parts = []
    profile = raw.get("profile") or {}

    # 1. Current position & domain
    title = profile.get("current_title", "")
    industry = profile.get("current_industry", "")
    if title and industry:
        parts.append(f"Currently working as a {title} in the {industry} industry.")
    elif title:
        parts.append(f"Currently working as a {title}.")

    # 2. Years of experience
    yoe = profile.get("years_of_experience")
    if yoe is not None:
        parts.append(f"Has {yoe} years of professional experience.")

    # 3. Technical skills — cap at 12 to keep text compact for fast encoding
    skills_raw = raw.get("skills") or []
    skill_names = []
    for s in skills_raw:
        if isinstance(s, dict):
            name = (s.get("name") or "").strip()
            if name:
                skill_names.append(name)
    if skill_names:
        parts.append(f"Skills: {', '.join(skill_names[:12])}.")

    # 4. Previous career titles (non-current roles)
    career = raw.get("career_history") or []
    prev_titles = []
    for role in career:
        if isinstance(role, dict) and not role.get("is_current"):
            t = role.get("title", "")
            if t:
                prev_titles.append(t)
    if prev_titles:
        parts.append(f"Previous: {', '.join(prev_titles)}.")

    # 5. Headline (compact signal — already a short string)
    headline = profile.get("headline", "")
    if headline:
        parts.append(headline)

    return " ".join(parts)


# ===========================================================================
# Main Precomputation Pipeline
# ===========================================================================
def precompute_embeddings(candidates_path: str, jd_path: str,
                          output_dir: str, batch_size: int = 64):
    """
    Reads all candidates from JSONL/JSONL.GZ, builds text profiles, encodes them
    using sentence-transformers, and saves to disk.
    """
    t_start = time.time()

    # ── 1. Load JD ──────────────────────────────────────────────────
    print(f"[1/5] Loading JD from {jd_path}...")
    with open(jd_path, "r", encoding="utf-8") as f:
        jd_data = json.load(f)

    jd_text = build_jd_profile_text(jd_data)
    print(f"    JD text: {jd_text[:100]}...")

    # ── 2. Build candidate texts ────────────────────────────────────
    print(f"[2/5] Building candidate profile texts from {candidates_path}...")
    candidate_texts = []
    candidate_ids = []

    with _open_candidates_file(candidates_path) as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                cid = raw.get("candidate_id", f"UNKNOWN_{idx}")
                text = build_candidate_profile_text(raw)

                candidate_ids.append(cid)
                candidate_texts.append(text)

                if (idx + 1) % 10000 == 0:
                    print(f"    Read {idx + 1:,} candidates...")
            except json.JSONDecodeError:
                print(f"    WARNING: Failed to parse line {idx + 1}, skipping.")
                continue

    total = len(candidate_ids)
    print(f"    Total: {total:,} candidate profiles built.")

    # ── 3. Load model ───────────────────────────────────────────────
    print(f"[3/5] Loading SentenceTransformer model: {MODEL_NAME}...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)
    # Use the new method name (get_embedding_dimension) if available,
    # fall back to deprecated name for older versions
    try:
        embedding_dim = model.get_embedding_dimension()
    except AttributeError:
        embedding_dim = model.get_sentence_embedding_dimension()
    print(f"    Model loaded. Embedding dim: {embedding_dim}")

    # ── 4. Encode ───────────────────────────────────────────────────
    print(f"[4/5] Encoding {total:,} candidate profiles (batch_size={batch_size})...")
    t_encode_start = time.time()

    # Encode JD first
    jd_embedding = model.encode(jd_text, convert_to_numpy=True).astype(np.float32)

    # Encode candidates in batches
    # sentence-transformers handles batching internally, but we use
    # show_progress_bar for visibility
    cand_embeddings = model.encode(
        candidate_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    t_encode_elapsed = time.time() - t_encode_start
    print(f"    Encoding complete in {t_encode_elapsed:.1f}s "
          f"({total / t_encode_elapsed:.0f} candidates/sec)")

    # Convert to float16 to save disk space (~73 MB vs 146 MB for float32)
    cand_embeddings_f16 = cand_embeddings.astype(np.float16)

    # ── 5. Save to disk ────────────────────────────────────────────
    print(f"[5/5] Saving to {output_dir}/...")
    os.makedirs(output_dir, exist_ok=True)

    emb_path = os.path.join(output_dir, "embeddings.npy")
    jd_emb_path = os.path.join(output_dir, "jd_embedding.npy")
    ids_path = os.path.join(output_dir, "candidate_ids.json")

    np.save(emb_path, cand_embeddings_f16)
    np.save(jd_emb_path, jd_embedding)
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(candidate_ids, f)

    emb_size_mb = os.path.getsize(emb_path) / (1024 * 1024)
    total_elapsed = time.time() - t_start

    print(f"\n{'='*60}")
    print(f"  PRECOMPUTATION COMPLETE")
    print(f"  Candidates encoded: {total:,}")
    print(f"  Embedding shape: {cand_embeddings_f16.shape} ({cand_embeddings_f16.dtype})")
    print(f"  Files saved:")
    print(f"    {emb_path} ({emb_size_mb:.1f} MB)")
    print(f"    {jd_emb_path}")
    print(f"    {ids_path}")
    print(f"  Total time: {total_elapsed:.1f}s")
    print(f"{'='*60}")


# ===========================================================================
# CLI Entry Point
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="TalentLens Bharat — Offline Embedding Precomputation"
    )
    parser.add_argument(
        "--candidates", "-c",
        default=os.path.join(BASE_DIR, "candidates.jsonl"),
        help="Path to candidates JSONL or JSONL.GZ file"
    )
    parser.add_argument(
        "--jd",
        default=os.path.join(BASE_DIR, "data", "parsed_jd.json"),
        help="Path to parsed JD JSON"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=os.path.join(BASE_DIR, "precomputed"),
        help="Directory to save precomputed files"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=64,
        help="Encoding batch size (default: 64)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.candidates):
        print(f"ERROR: Candidates file not found: {args.candidates}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.jd):
        print(f"ERROR: Parsed JD file not found: {args.jd}", file=sys.stderr)
        print(f"       Create data/parsed_jd.json first.", file=sys.stderr)
        sys.exit(1)

    precompute_embeddings(
        candidates_path=args.candidates,
        jd_path=args.jd,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
