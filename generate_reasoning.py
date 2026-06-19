#!/usr/bin/env python3
"""
TalentLens Bharat — generate_reasoning.py
==========================================
Offline LLM reasoning cache generator.

Run this ONCE after ranking to pre-generate LLM-powered reasoning
strings for the top 150 candidates (buffer above the 100 needed for CSV).

Output:
  data/reasoning_cache.json  →  {"CAND_XXXXXXX": "reasoning string", ...}

Usage:
  python generate_reasoning.py
  python generate_reasoning.py --top-n 200 --model llama-3.1-8b-instant
  python generate_reasoning.py --candidates candidates.jsonl.gz

This script DOES make network calls (Groq API). The output JSON is used by
rank.py which has ZERO network dependencies.

Expected runtime: ~3-5 minutes for 150 candidates (rate-limited API calls).
"""

import argparse
import gzip
import json
import os
import sys
import time

import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ===========================================================================
# File Opener Helper
# ===========================================================================
def _open_candidates_file(path: str):
    """Returns an appropriate file handle for .jsonl or .jsonl.gz files."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    else:
        return open(path, "r", encoding="utf-8")


# ===========================================================================
# Quick Offline Ranking (to find top candidates)
# ===========================================================================
def quick_rank_top_n(candidates_path: str, precomputed_dir: str,
                     jd_path: str, top_n: int = 150) -> list:
    """
    Performs full ranking using rank.py's scoring logic to identify the true
    top N candidates for reasoning generation.

    Returns a list of (candidate_id, semantic_score, raw_candidate_dict) tuples.
    """
    import rank
    import datetime

    # Load JD
    with open(jd_path, "r", encoding="utf-8") as f:
        jd_data = json.load(f)

    jd_seniority = jd_data.get("seniority", "Senior")
    jd_required_skills = jd_data.get("required_skills", [])
    jd_nice_to_have = jd_data.get("nice_to_have", [])

    # Load precomputed embeddings
    emb_path = os.path.join(precomputed_dir, "embeddings.npy")
    jd_emb_path = os.path.join(precomputed_dir, "jd_embedding.npy")
    ids_path = os.path.join(precomputed_dir, "candidate_ids.json")

    has_embeddings = all(os.path.exists(p) for p in [emb_path, jd_emb_path, ids_path])
    if has_embeddings:
        print("  Loading precomputed embeddings...")
        embeddings = np.load(emb_path)
        jd_embedding = np.load(jd_emb_path)
        with open(ids_path, "r") as f:
            precomputed_ids = json.load(f)
        id_to_idx = {cid: idx for idx, cid in enumerate(precomputed_ids)}

        if embeddings.dtype == np.float16:
            embeddings = embeddings.astype(np.float32)
        emb_norms = np.linalg.norm(embeddings, axis=1)
        jd_norm = np.linalg.norm(jd_embedding)
        emb_norms[emb_norms == 0] = 1.0
        if jd_norm == 0:
            jd_norm = 1.0
        all_semantic_scores = embeddings.dot(jd_embedding) / (emb_norms * jd_norm)
    else:
        id_to_idx = {}
        all_semantic_scores = None

    today = datetime.date.today()
    scored_candidates = []

    print("  Streaming candidates to compute true ranks...")
    with _open_candidates_file(candidates_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue

            candidate_id = raw.get("candidate_id", "")

            # Honeypot check
            if rank.is_honeypot(raw):
                scored_candidates.append((candidate_id, 0.0500, 0.0500, raw))
                continue

            feat = rank.extract_candidate_features(raw)

            # Semantic score
            if all_semantic_scores is not None and candidate_id in id_to_idx:
                semantic_score = float(all_semantic_scores[id_to_idx[candidate_id]])
            else:
                semantic_score = rank._keyword_fallback_score(feat, jd_required_skills, jd_nice_to_have)

            seniority_mult = rank.compute_seniority_penalty(jd_seniority, feat["current_title"])
            india_score = rank.compute_india_platform_score(feat, today)
            trajectory_score = rank.compute_trajectory_score(feat)
            behavioral_score = rank.compute_behavioral_signal_score(feat, today)
            disqualifier_mult = rank.compute_disqualifier_penalty(feat)
            stuffer_mult = rank.compute_honeypot_penalty(feat)

            final_score = disqualifier_mult * stuffer_mult * (
                rank.W_SEMANTIC * semantic_score * seniority_mult
                + rank.W_INDIA * india_score
                + rank.W_TRAJECTORY * trajectory_score
                + rank.W_BEHAVIORAL * behavioral_score
            )
            final_score = max(0.0, min(1.0, final_score))
            final_score = round(final_score, 4)

            scored_candidates.append((candidate_id, final_score, semantic_score, raw))

    # Sort candidates by final_score descending, candidate_id ascending
    scored_candidates.sort(key=lambda x: (-x[1], x[0]))

    # Return top N as (candidate_id, semantic_score, raw)
    top_candidates = [(cid, sem_score, raw) for cid, final_score, sem_score, raw in scored_candidates[:top_n]]

    print(f"  True top {top_n} candidates identified by final rank score.")
    return top_candidates


# ===========================================================================
# Reasoning Generation via LLM
# ===========================================================================
def generate_reasoning_for_candidate(llm, jd_data: dict, raw_candidate: dict,
                                     semantic_score: float) -> str:
    """
    Calls the LLM to generate a concise, recruiter-facing reasoning string
    for why this candidate is a match (or not) for the JD.

    Returns a single-paragraph reasoning string.
    """
    profile = raw_candidate.get("profile") or {}
    skills_raw = raw_candidate.get("skills") or []
    career_raw = raw_candidate.get("career_history") or []
    signals = raw_candidate.get("redrob_signals") or {}

    # Build context
    skill_names = [s.get("name", "") for s in skills_raw if isinstance(s, dict)]
    career_titles = [r.get("title", "") for r in career_raw if isinstance(r, dict)]

    candidate_summary = (
        f"Name: {profile.get('anonymized_name', 'Unknown')}\n"
        f"Title: {profile.get('current_title', 'N/A')}\n"
        f"Industry: {profile.get('current_industry', 'N/A')}\n"
        f"Experience: {profile.get('years_of_experience', 'N/A')} years\n"
        f"Location: {profile.get('location', 'N/A')}, {profile.get('country', 'N/A')}\n"
        f"Skills: {', '.join(skill_names[:15])}\n"
        f"Career History: {' → '.join(career_titles)}\n"
        f"GitHub Activity: {signals.get('github_activity_score', 'N/A')}\n"
        f"Recruiter Response Rate: {signals.get('recruiter_response_rate', 'N/A')}\n"
        f"Open to Work: {signals.get('open_to_work_flag', 'N/A')}\n"
        f"Semantic Match Score: {semantic_score:.4f}"
    )

    jd_summary = (
        f"Role: {jd_data.get('seniority', 'Senior')} level in {jd_data.get('domain', 'AI/ML Engineering')}\n"
        f"Required Skills: {', '.join(jd_data.get('required_skills', []))}\n"
        f"Nice to Have: {', '.join(jd_data.get('nice_to_have', []))}"
    )

    prompt = (
        "You are an expert recruiter writing a brief, factual justification for why this candidate is ranked in the top matching profiles. "
        "Write exactly ONE sentence (max 120 words) highlighting their key alignment (such as experience, title, domain, or relevant skills) and constructively framing any minor gaps. "
        "Focus on justifying their high rank. Do NOT use subjective adjectives like 'excellent' or 'strong' — use specific facts and numbers.\n\n"
        f"--- JOB DESCRIPTION ---\n{jd_summary}\n\n"
        f"--- CANDIDATE ---\n{candidate_summary}\n\n"
        "Reasoning (one sentence):"
    )


    try:
        response = llm.invoke(prompt)
        text = response.content.strip()
        # Clean up: remove quotes, limit length
        text = text.strip('"').strip("'")
        if len(text) > 400:
            text = text[:397] + "..."
        return text
    except Exception as e:
        print(f"    LLM error: {e}")
        return None


def generate_fallback_reasoning(raw_candidate: dict, jd_data: dict) -> str:
    """
    Programmatic fallback reasoning when LLM is unavailable.
    """
    profile = raw_candidate.get("profile") or {}
    skills_raw = raw_candidate.get("skills") or []
    signals = raw_candidate.get("redrob_signals") or {}

    title = profile.get("current_title", "Professional")
    yoe = profile.get("years_of_experience", 0)
    industry = profile.get("current_industry", "")
    rr = signals.get("recruiter_response_rate", 0)

    # Count skills matching JD
    skill_names_lower = [
        s.get("name", "").lower() for s in skills_raw if isinstance(s, dict)
    ]
    jd_skills = jd_data.get("required_skills", [])
    matched = sum(
        1 for js in jd_skills
        if any(js.lower() in cs or cs in js.lower() for cs in skill_names_lower)
    )

    parts = [f"{title} with {yoe:.1f} yrs"]
    parts.append(f"{matched}/{len(jd_skills)} required skills matched")
    if industry:
        parts.append(f"{industry} background")
    parts.append(f"response rate {rr:.2f}")

    return "; ".join(parts) + "."


# ===========================================================================
# Main Generation Pipeline
# ===========================================================================
def generate_reasoning_cache(candidates_path: str, jd_path: str,
                             precomputed_dir: str, output_path: str,
                             top_n: int = 150, model_name: str = "llama-3.1-8b-instant"):
    """
    Main entry point. Identifies top candidates, generates reasoning, saves cache.
    """
    t_start = time.time()

    # ── 1. Load JD ──────────────────────────────────────────────────
    print(f"[1/4] Loading JD from {jd_path}...")
    with open(jd_path, "r", encoding="utf-8") as f:
        jd_data = json.load(f)

    # ── 2. Quick rank to find top N ─────────────────────────────────
    print(f"[2/4] Quick-ranking to find top {top_n} candidates...")
    top_candidates = quick_rank_top_n(
        candidates_path, precomputed_dir, jd_path, top_n
    )
    print(f"    Found {len(top_candidates)} candidates for reasoning generation.")

    # ── 3. Initialize LLM ──────────────────────────────────────────
    print(f"[3/4] Initializing LLM ({model_name})...")
    try:
        from dotenv import load_dotenv
        load_dotenv()

        api_key = os.environ.get("GROQ_API_KEY", "")

        if api_key.startswith("xai-"):
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model="grok-beta",
                temperature=0.0,
                api_key=api_key,
                base_url="https://api.x.ai/v1",
                max_retries=2,
            )
            print("    Using xAI (Grok) API.")
        else:
            from langchain_groq import ChatGroq
            llm = ChatGroq(
                model=model_name,
                temperature=0.0,
                api_key=api_key,
                max_retries=2,
            )
            print("    Using Groq API.")
    except Exception as e:
        print(f"    WARNING: Failed to initialize LLM: {e}")
        print(f"    Will generate programmatic fallback reasoning only.")
        llm = None

    # ── 4. Generate reasoning for each candidate ────────────────────
    print(f"[4/4] Generating reasoning for {len(top_candidates)} candidates...")
    
    # Load existing cache if available
    existing_cache = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing_cache = json.load(f)
            print(f"    Loaded {len(existing_cache)} existing reasoning entries to avoid redundant API calls.")
        except Exception as e:
            print(f"    WARNING: Failed to load existing cache: {e}")

    reasoning_cache = {}
    llm_success = 0
    fallback_count = 0
    reused_count = 0

    for i, (cid, score, raw) in enumerate(top_candidates):
        # Try to reuse LLM-generated reasoning from existing cache
        reasoning = existing_cache.get(cid)
        if reasoning and "required skills matched" not in reasoning and not reasoning.startswith("Profile flagged:"):
            reasoning_cache[cid] = reasoning
            reused_count += 1
            if (i + 1) % 10 == 0:
                elapsed = time.time() - t_start
                print(f"    [{i+1}/{len(top_candidates)}] "
                      f"LLM: {llm_success}, Reused: {reused_count}, Fallback: {fallback_count} "
                      f"({elapsed:.1f}s)")
            continue

        # Try LLM first
        reasoning = None
        if llm is not None:
            reasoning = generate_reasoning_for_candidate(llm, jd_data, raw, score)
            if reasoning:
                llm_success += 1

        # Fallback if LLM failed
        if not reasoning:
            reasoning = generate_fallback_reasoning(raw, jd_data)
            fallback_count += 1

        reasoning_cache[cid] = reasoning

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t_start
            print(f"    [{i+1}/{len(top_candidates)}] "
                  f"LLM: {llm_success}, Reused: {reused_count}, Fallback: {fallback_count} "
                  f"({elapsed:.1f}s)")

        # Rate limiting: small delay between LLM calls
        if llm is not None and reasoning and not reasoning.startswith("Profile flagged:") and "required skills matched" not in reasoning:
            time.sleep(0.2)  # 200ms between calls to avoid rate limits

    # ── Save cache ──────────────────────────────────────────────────
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(reasoning_cache, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  REASONING GENERATION COMPLETE")
    print(f"  Total candidates: {len(reasoning_cache)}")
    print(f"  LLM-generated: {llm_success}")
    print(f"  Reused from cache: {reused_count}")
    print(f"  Programmatic fallback: {fallback_count}")
    print(f"  Saved to: {output_path}")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"{'='*60}")


# ===========================================================================
# CLI Entry Point
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="TalentLens Bharat — Offline Reasoning Cache Generator"
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
        "--precomputed-dir",
        default=os.path.join(BASE_DIR, "precomputed"),
        help="Path to precomputed embeddings directory"
    )
    parser.add_argument(
        "--out", "-o",
        default=os.path.join(BASE_DIR, "data", "reasoning_cache.json"),
        help="Output path for reasoning cache JSON"
    )
    parser.add_argument(
        "--top-n", "-n",
        type=int,
        default=150,
        help="Number of top candidates to generate reasoning for (default: 150)"
    )
    parser.add_argument(
        "--model", "-m",
        default="llama-3.1-8b-instant",
        help="LLM model name for Groq (default: llama-3.1-8b-instant)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.candidates):
        print(f"ERROR: Candidates file not found: {args.candidates}", file=sys.stderr)
        sys.exit(1)

    generate_reasoning_cache(
        candidates_path=args.candidates,
        jd_path=args.jd,
        precomputed_dir=args.precomputed_dir,
        output_path=args.out,
        top_n=args.top_n,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()
