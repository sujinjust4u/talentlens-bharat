#!/usr/bin/env python3
"""
TalentLens Bharat — rank.py
============================
Production ranking script for the Redrob AI Hackathon submission.

Constraints (hard requirements from submission_spec):
  ✗ ZERO network calls — no API, no LLM, no HTTP
  ✗ Must process 100,000 candidates in < 5 minutes on CPU (16 GB RAM)
  ✗ Output: CSV with exactly (candidate_id, rank, score, reasoning), 100 data rows
  ✗ Scores must be non-increasing; ties broken by candidate_id ascending
  ✗ Honeypot profiles (impossible skills) must not appear in top 100

Usage:
  python rank.py --candidates ./candidates.jsonl --out ./submission.csv
  python rank.py --candidates ./candidates.jsonl.gz --out ./submission.csv

Dependencies (all offline/local):
  - numpy (vectorized cosine similarity)
  - json, csv, gzip, argparse (stdlib)

Pre-requirements (run ONCE before submission):
  - python precompute.py     → precomputed/embeddings.npy, jd_embedding.npy, candidate_ids.json
  - python generate_reasoning.py → data/reasoning_cache.json

Expected runtime: ~30-90 seconds for 100K candidates on M2 MacBook (16 GB).
Expected memory: ~600 MB peak (embeddings matrix + candidate dicts in flight).
"""

import argparse
import csv
import gzip
import json
import os
import sys
import time
from datetime import date, datetime


# ===========================================================================
# IMPORTANT: Only stdlib + numpy. NO langchain, NO sentence_transformers,
# NO openai, NO groq, NO requests, NO httpx. This file must be network-free.
# ===========================================================================
import numpy as np


# ===========================================================================
# Section 0: Constants & Configuration
# ===========================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Scoring weights (documented in README) ──────────────────────────────
# These four components sum to 1.0 and reflect the relative importance
# of each signal type for the Senior AI Engineer role.
W_SEMANTIC   = 0.55   # Cosine similarity of profile embedding vs JD embedding
W_INDIA      = 0.20   # Platform activity + India-specific signals
W_TRAJECTORY = 0.10   # Career momentum (skill velocity, title progression)
W_BEHAVIORAL = 0.15   # Redrob platform behavioral signals (open_to_work, recency, etc.)

# ── Seniority level mapping (0=intern → 5=principal/director) ───────────
SENIORITY_LEVELS = {
    "intern": 0, "trainee": 0,
    "junior": 1, "associate": 1, "entry": 1,
    "mid": 2, "middle": 2,
    "senior": 3, "sr": 3,
    "lead": 4, "team lead": 4, "tech lead": 4,
    "principal": 5, "architect": 5, "staff": 5, "director": 5, "vp": 5,
}

# ── Skills classified by seniority (for trajectory scoring) ─────────────
SENIOR_LEAD_SKILLS = {
    "kubernetes", "langchain", "llms", "large language models", "system design",
    "vector databases", "kafka", "terraform", "microservices", "grpc",
    "hugging face", "huggingface", "pytorch", "airflow", "ml ops", "mlops",
    "fine-tuning", "fine tuning", "fine-tuning llms", "rag", "rag pipelines",
    "distributed systems", "platform architecture", "mlflow", "model serving",
    "feature stores", "kubeflow", "ray", "dask", "spark streaming",
    "technical roadmap", "scikit-learn", "sklearn", "deep learning",
    "reinforcement learning", "computer vision", "nlp",
    "natural language processing", "transformer", "transformers",
    "prompt engineering", "embeddings", "vector search", "faiss",
    "pinecone", "qdrant", "milvus", "weaviate", "chroma",
    "sentence-transformers", "bge", "e5", "openai embeddings",
    "lora", "qlora", "peft", "learning-to-rank", "ndcg", "mrr",
}

MID_SKILLS = {
    "docker", "postgresql", "postgres", "redis", "fastapi", "django",
    "flask", "aws", "azure", "gcp", "ci/cd", "cicd", "sqlalchemy",
    "pytest", "celery", "mongodb", "elasticsearch", "graphql",
    "rest api", "rest apis", "typescript", "react", "node.js", "nodejs",
}

JUNIOR_SKILLS = {
    "git", "html", "css", "basic python", "mysql", "sql", "excel",
    "jira", "basic sql", "javascript", "bootstrap",
}

# ── Tier-2/3 Indian cities ──────────────────────────────────────────────
TIER2_3_CITIES = {
    "indore", "bhopal", "nagpur", "coimbatore", "surat", "jaipur",
    "lucknow", "kochi", "cochin", "chandigarh", "vadodara", "madurai",
    "tiruchirappalli", "trichy", "salem", "nashik", "agra", "varanasi",
    "patna", "ranchi", "bhubaneswar", "thiruvananthapuram", "trivandrum",
    "visakhapatnam", "vizag", "mysore", "mysuru", "mangalore", "mangaluru",
    "dehradun", "amritsar", "jodhpur", "udaipur", "raipur", "guwahati",
    "allahabad", "prayagraj", "kanpur", "meerut", "ludhiana", "jalandhar",
    "aurangabad", "rajkot", "vijayawada", "warangal", "guntur",
    "tiruvallur", "kozhikode", "calicut", "thrissur", "hubli",
    "belgaum", "belagavi", "gulbarga", "kalaburagi", "shimla",
    "pondicherry", "puducherry", "siliguri", "durgapur", "asansol",
    "jamshedpur", "dhanbad", "bokaro", "cuttack", "rourkela",
    "nellore", "tirupati", "anantapur", "kakinada", "rajahmundry",
    "bareilly", "aligarh", "moradabad", "gorakhpur", "mathura",
    "firozabad", "gwalior", "ujjain", "jabalpur", "kolhapur",
    "solapur", "sangli", "nanded", "latur", "akola", "amravati",
    "bharuch", "anand", "gandhidham", "bhavnagar", "junagadh",
}

# ── Tier-1 Indian cities ────────────────────────────────────────────────
TIER1_CITIES = {
    "mumbai", "delhi", "bangalore", "bengaluru", "hyderabad",
    "chennai", "kolkata", "pune", "ahmedabad", "noida",
    "gurgaon", "gurugram", "new delhi", "navi mumbai", "thane",
}

# ── India-specific technologies ─────────────────────────────────────────
INDIA_TECH_KEYWORDS = {
    "tally", "upi", "bhim", "rupay", "fastag", "aadhaar", "gstin",
    "digilocker", "paytm", "razorpay", "phonepe",
}

# ── Consulting firms (disqualifier if entire career) ────────────────────
# Candidates with ALL career_history at these firms and no product-company
# experience get a scoring penalty.
CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "tata consultancy services",
    "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "hcl technologies", "tech mahindra",
    "mindtree", "mphasis", "l&t infotech", "ltimindtree",
    "hexaware", "cyient", "zensar", "persistent systems",
}


# ===========================================================================
# Section 1: Feature Extraction from Real Schema
# ===========================================================================
def extract_candidate_features(raw: dict) -> dict:
    """
    Extracts a flat feature dict from the nested candidate schema.
    Handles missing/null fields gracefully at every level.
    Supports both nested (real dataset) and flat (mock/PM dataset) structures.
    """
    profile = raw.get("profile") or {}
    signals = raw.get("redrob_signals") or {}
    skills_raw = raw.get("skills") or []
    career_raw = raw.get("career_history") or []
    education_raw = raw.get("education") or []

    # ── Extract skill names and metadata ────────────────────────────
    skill_names = []
    skill_proficiencies = {}
    skill_endorsements = {}
    skill_durations = {}
    total_endorsements = 0

    for s in skills_raw:
        if isinstance(s, str):
            name = s.strip()
            if not name:
                continue
            name_lower = name.lower()
            skill_names.append(name)
            skill_proficiencies[name_lower] = "intermediate"
            skill_endorsements[name_lower] = 0
            skill_durations[name_lower] = 12  # Default to 12 months for flat/mock skills
        elif isinstance(s, dict):
            name = (s.get("name") or "").strip()
            if not name:
                continue
            name_lower = name.lower()
            skill_names.append(name)
            skill_proficiencies[name_lower] = s.get("proficiency", "intermediate")
            skill_endorsements[name_lower] = s.get("endorsements", 0)
            skill_durations[name_lower] = s.get("duration_months", 0)
            total_endorsements += s.get("endorsements", 0) or 0

    # ── Career history ──────────────────────────────────────────────
    career_titles = []
    career_companies = []
    career_durations = []
    
    if career_raw:
        for role in career_raw:
            if not isinstance(role, dict):
                continue
            title = role.get("title", "")
            company = role.get("company", "")
            duration = role.get("duration_months", 0) or 0
            if title:
                career_titles.append(title)
            if company:
                career_companies.append(company)
            career_durations.append(duration)
    else:
        # Fallback to flat schema previous_titles and current_title
        prev_titles = raw.get("previous_titles") or []
        for pt in prev_titles:
            if isinstance(pt, str) and pt:
                career_titles.append(pt)
        curr_t = raw.get("current_title")
        if curr_t and curr_t not in career_titles:
            career_titles.insert(0, curr_t)

    # ── Education tiers ─────────────────────────────────────────────
    edu_tiers = []
    for edu in education_raw:
        if isinstance(edu, dict):
            edu_tiers.append(edu.get("tier", "unknown"))

    # ── Skill assessment scores from Redrob platform ────────────────
    assessment_scores = signals.get("skill_assessment_scores") or {}

    # Extract ID
    cand_id = raw.get("candidate_id") or raw.get("id") or ""

    return {
        "candidate_id": cand_id,
        "name": profile.get("anonymized_name") or raw.get("name") or "Unknown",
        "headline": profile.get("headline") or raw.get("bio") or raw.get("headline") or "",
        "summary": profile.get("summary") or raw.get("bio") or raw.get("summary") or "",
        "location": profile.get("location") or raw.get("location") or "",
        "country": profile.get("country") or raw.get("country") or "",
        "years_of_experience": profile.get("years_of_experience") or raw.get("experience_years") or raw.get("years_of_experience") or 0.0,
        "current_title": profile.get("current_title") or raw.get("current_title") or "",
        "current_industry": profile.get("current_industry") or raw.get("domain") or raw.get("current_industry") or "",
        "current_company_size": profile.get("current_company_size") or raw.get("current_company_size") or "",
        "current_company": profile.get("current_company") or raw.get("current_company") or "",

        # Skills
        "skill_names": skill_names,
        "skill_names_lower": [s.lower() for s in skill_names],
        "skill_proficiencies": skill_proficiencies,
        "skill_endorsements": skill_endorsements,
        "skill_durations": skill_durations,
        "total_endorsements": total_endorsements,
        "num_skills": len(skill_names),

        # Career
        "career_titles": career_titles,
        "career_companies": career_companies,
        "career_durations": career_durations,
        "num_career_roles": len(career_titles),

        # Education
        "edu_tiers": edu_tiers,

        # Redrob platform signals (raw values for behavioral scoring)
        "last_active_date": signals.get("last_active_date") or raw.get("last_active_date") or "",
        "open_to_work": signals.get("open_to_work_flag") or raw.get("open_to_work") or False,
        "profile_completeness": signals.get("profile_completeness_score") or raw.get("profile_completeness") or 0.0,
        "recruiter_response_rate": signals.get("recruiter_response_rate") or raw.get("recruiter_response_rate") or 0.0,
        "avg_response_time_hours": signals.get("avg_response_time_hours") or raw.get("avg_response_time_hours") or 999.0,
        "github_activity_score": signals.get("github_activity_score") or raw.get("github_activity_score") or raw.get("github_activity") or -1.0,
        "interview_completion_rate": signals.get("interview_completion_rate") or raw.get("interview_completion_rate") or 0.0,
        "offer_acceptance_rate": signals.get("offer_acceptance_rate") or raw.get("offer_acceptance_rate") or -1.0,
        "notice_period_days": signals.get("notice_period_days") or raw.get("notice_period_days") or 90.0,
        "willing_to_relocate": signals.get("willing_to_relocate") or raw.get("willing_to_relocate") or False,
        "preferred_work_mode": signals.get("preferred_work_mode") or raw.get("preferred_work_mode") or "onsite",
        "saved_by_recruiters_30d": signals.get("saved_by_recruiters_30d") or raw.get("saved_by_recruiters_30d") or 0.0,
        "profile_views_30d": signals.get("profile_views_received_30d") or raw.get("profile_views_30d") or 0.0,
        "verified_email": signals.get("verified_email") or raw.get("verified_email") or False,
        "verified_phone": signals.get("verified_phone") or raw.get("verified_phone") or False,
        "assessment_scores": assessment_scores,
    }


# ===========================================================================
# Section 2: Honeypot Detection
# ===========================================================================
def is_honeypot(raw: dict) -> bool:
    """
    Detects honeypot/impossible profiles planted in the dataset.
    
    Honeypot indicators (any one triggers):
      1. Skill with proficiency "advanced" or "expert" but duration_months = 0
         → Impossible: you can't be expert in something you've never used
      2. 10+ skills all at "expert" level
         → Statistically implausible
    
    Returns True if candidate is a honeypot.
    """
    skills = raw.get("skills") or []
    expert_count = 0

    for skill in skills:
        if not isinstance(skill, dict):
            continue
        proficiency = skill.get("proficiency", "")
        duration = skill.get("duration_months", 1)  # Default 1 to avoid false positives

        # Check 1: Expert/advanced with zero duration
        if proficiency in ("advanced", "expert"):
            if duration == 0:
                return True
            expert_count += 1

    # Check 2: Unrealistically many expert skills
    if expert_count >= 10:
        return True

    return False


# ===========================================================================
# Section 3: Seniority Mapping and Penalty
# ===========================================================================
def _infer_seniority_level(title: str) -> int:
    """
    Maps a job title string to a seniority integer (0-5).
    Uses keyword matching on the title. Defaults to 2 (mid-level).
    """
    if not title:
        return 2
    t = title.lower()

    # Check from highest to lowest to avoid false matches
    if any(kw in t for kw in ("director", "vp ", "vice president")):
        return 5
    if any(kw in t for kw in ("principal", "architect", "staff", "distinguished")):
        return 5
    if any(kw in t for kw in ("lead", "head ", "head of", "team lead", "tech lead", "manager")):
        return 4
    if any(kw in t for kw in ("senior", "sr.", "sr ")):
        return 3
    if any(kw in t for kw in ("junior", "jr.", "jr ", "trainee", "intern", "apprentice")):
        return 1
    if any(kw in t for kw in ("entry", "fresher", "graduate")):
        return 1

    return 2  # Default mid


def compute_seniority_penalty(jd_seniority: str, candidate_title: str) -> float:
    """
    Computes a multiplier (0.70 – 1.0) based on the seniority gap
    between the JD requirement and the candidate's current title.

    Gap 0 → 1.00 (exact match)
    Gap 1 → 0.92 (close enough, slight penalty)
    Gap 2 → 0.82 (noticeable mismatch)
    Gap 3+ → 0.70 (severe mismatch — intern applying for senior role)
    """
    jd_level = _infer_seniority_level(jd_seniority)
    cand_level = _infer_seniority_level(candidate_title)
    gap = abs(jd_level - cand_level)

    if gap == 0:
        return 1.00
    elif gap == 1:
        return 0.92
    elif gap == 2:
        return 0.82
    else:
        return 0.70


# ===========================================================================
# Section 4: India Platform Signal Scoring
# ===========================================================================
def compute_india_platform_score(feat: dict, today: date = None) -> float:
    """
    Computes a platform + India-relevance signal score (0.0 – 1.0).

    Components and their max contributions:
      1. Activity recency:        max +0.15
      2. Open to work:            +0.06
      3. Profile completeness:    max +0.06
      4. Recruiter responsiveness: max +0.05
      5. GitHub activity:         max +0.04
      6. Interview reliability:   +0.03
      7. India country & city:    max +0.12
      8. Education tier:          max +0.06
      9. Verified profile:        +0.02
     10. Null skills penalty:     -0.15
     11. Saved by recruiters:     max +0.03

    Base score: 0.35 → range [0.20, 1.0]
    """
    if today is None:
        today = date.today()

    score = 0.35  # Base score

    # ── 1. Activity recency (max +0.15) ─────────────────────────────
    last_active_str = feat.get("last_active_date", "")
    days_inactive = None
    if last_active_str:
        try:
            last_active = datetime.strptime(str(last_active_str).strip(), "%Y-%m-%d").date()
            days_inactive = max(0, (today - last_active).days)
        except (ValueError, TypeError):
            pass

    if days_inactive is not None:
        if days_inactive < 14:
            score += 0.15
        elif days_inactive < 30:
            score += 0.12
        elif days_inactive < 60:
            score += 0.06
        elif days_inactive < 90:
            score += 0.03
        elif days_inactive > 180:
            score -= 0.05

    # ── 2. Open to work (+0.06) ─────────────────────────────────────
    if feat.get("open_to_work"):
        score += 0.06

    # ── 3. Profile completeness (max +0.06) ─────────────────────────
    completeness = feat.get("profile_completeness", 0) or 0
    if completeness >= 85:
        score += 0.06
    elif completeness >= 70:
        score += 0.03

    # ── 4. Recruiter responsiveness (+0.05) ─────────────────────────
    rr = feat.get("recruiter_response_rate", 0) or 0
    if rr >= 0.6:
        score += 0.05
    elif rr >= 0.3:
        score += 0.02

    # ── 5. GitHub activity (max +0.04) ──────────────────────────────
    github = feat.get("github_activity_score", -1)
    if github is not None and github >= 0:
        if github >= 60:
            score += 0.04
        elif github >= 30:
            score += 0.02

    # ── 6. Interview reliability (+0.03) ────────────────────────────
    icr = feat.get("interview_completion_rate", 0) or 0
    if icr >= 0.7:
        score += 0.03

    # ── 7. India country & city tier (max +0.12) ────────────────────
    country = (feat.get("country") or "").strip().lower()
    location = (feat.get("location") or "").strip().lower()

    is_india = country in ("india", "in")
    if is_india:
        score += 0.04  # India relevance bonus

        # Tier-2/3 city detection → strong representation boost
        for city in TIER2_3_CITIES:
            if city in location:
                score += 0.08
                break

    # ── 8. Education tier (max +0.06) ───────────────────────────────
    edu_tiers = feat.get("edu_tiers", [])
    best_tier = "unknown"
    for t in edu_tiers:
        if t == "tier_1":
            best_tier = "tier_1"
            break
        elif t == "tier_2" and best_tier != "tier_1":
            best_tier = "tier_2"

    if best_tier == "tier_1":
        score += 0.06
    elif best_tier == "tier_2":
        score += 0.03

    # ── 9. Verified profile (+0.02) ─────────────────────────────────
    if feat.get("verified_email") and feat.get("verified_phone"):
        score += 0.02

    # ── 10. Null skills penalty (−0.15) ─────────────────────────────
    if feat.get("num_skills", 0) == 0:
        score -= 0.15

    # ── 11. Saved by recruiters (+0.03) ─────────────────────────────
    saved = feat.get("saved_by_recruiters_30d", 0) or 0
    if saved >= 5:
        score += 0.03
    elif saved >= 2:
        score += 0.01

    return max(0.0, min(1.0, score))


# ===========================================================================
# Section 5: Career Trajectory Scoring
# ===========================================================================
def compute_trajectory_score(feat: dict) -> float:
    """
    Computes career trajectory / momentum score (0.0 – 1.0).

    Components:
      1. Skill velocity (0.0–0.4): count of senior/lead-level skills
         above the candidate's current title seniority
      2. Title progression speed (0.0–0.3): number of career roles
         relative to years of experience
      3. Advanced recency (0.0–0.3): recently-acquired skills (duration < 12 months)
         that are at senior/lead level
    """
    skill_velocity = 0.0
    title_speed = 0.0
    advanced_recency = 0.0

    cand_level = _infer_seniority_level(feat.get("current_title", ""))
    yoe = feat.get("years_of_experience", 0) or 0

    # ── 1. Skill Velocity ───────────────────────────────────────────
    above_level_count = 0
    skill_names_lower = feat.get("skill_names_lower", [])

    for skill_lower in skill_names_lower:
        if cand_level < 3 and skill_lower in SENIOR_LEAD_SKILLS:
            above_level_count += 1
        elif cand_level < 2 and skill_lower in MID_SKILLS:
            above_level_count += 1

    if above_level_count >= 5:
        skill_velocity = 0.4
    elif above_level_count >= 4:
        skill_velocity = 0.35
    elif above_level_count >= 3:
        skill_velocity = 0.3
    elif above_level_count >= 2:
        skill_velocity = 0.2
    elif above_level_count >= 1:
        skill_velocity = 0.1

    # ── 2. Title Progression Speed ──────────────────────────────────
    num_roles = feat.get("num_career_roles", 0)
    if num_roles >= 3 and yoe > 0 and yoe < 5:
        title_speed = 0.3  # Fast mover: 3+ roles in < 5 years
    elif num_roles >= 3 and yoe > 0 and yoe < 8:
        title_speed = 0.2  # Solid progression
    elif num_roles >= 2 and yoe > 0 and yoe < 6:
        title_speed = 0.15
    elif num_roles >= 2:
        title_speed = 0.1  # Standard

    # ── 3. Advanced Recency ─────────────────────────────────────────
    recent_advanced_count = 0
    durations = feat.get("skill_durations", {})
    for skill_lower, duration in durations.items():
        if duration > 0 and duration < 12:
            if skill_lower in SENIOR_LEAD_SKILLS:
                recent_advanced_count += 1

    if recent_advanced_count >= 3:
        advanced_recency = 0.3
    elif recent_advanced_count >= 2:
        advanced_recency = 0.2
    elif recent_advanced_count >= 1:
        advanced_recency = 0.1

    raw = skill_velocity + title_speed + advanced_recency
    return max(0.0, min(1.0, raw))


# ===========================================================================
# Section 6: Behavioral Signal Scoring (NEW)
# ===========================================================================
def compute_behavioral_signal_score(feat: dict, today: date = None) -> float:
    """
    Computes a behavioral engagement score (0.0 – 1.0) from Redrob 
    platform signals that indicate candidate availability and reliability.

    Components:
      - open_to_work_flag:      +0.30 if True
      - last_active_date:       0.00–0.30 based on days since active
      - recruiter_response_rate: 0.00–0.25 (linear)
      - github_activity_score:  0.00–0.15 if score > 0

    This score measures "how likely is this person to actually respond
    and engage if we reach out?" — a critical recruiter signal.
    """
    if today is None:
        today = date.today()

    score = 0.0

    # ── 1. Open to work flag (+0.30) ────────────────────────────────
    # The single strongest behavioral signal: candidate has explicitly
    # declared they're looking for new opportunities.
    if feat.get("open_to_work"):
        score += 0.30

    # ── 2. Activity recency (0.00–0.30) ─────────────────────────────
    # How recently the candidate was active on the Redrob platform.
    # Inactive > 180 days → candidate likely not checking messages.
    last_active_str = feat.get("last_active_date", "")
    if last_active_str:
        try:
            last_active = datetime.strptime(str(last_active_str).strip(), "%Y-%m-%d").date()
            days_inactive = max(0, (today - last_active).days)

            if days_inactive < 7:
                score += 0.30     # Active this week
            elif days_inactive < 14:
                score += 0.25
            elif days_inactive < 30:
                score += 0.20
            elif days_inactive < 60:
                score += 0.12
            elif days_inactive < 90:
                score += 0.06
            elif days_inactive < 180:
                score += 0.03
            # > 180 days → 0.00 (ghost profile)
        except (ValueError, TypeError):
            pass

    # ── 3. Recruiter response rate (0.00–0.25) ──────────────────────
    # Direct linear contribution. A candidate who responds to 80% of
    # recruiter messages is far more valuable than one at 10%.
    rr = feat.get("recruiter_response_rate", 0) or 0
    score += min(0.25, rr * 0.25)

    # ── 4. GitHub activity score (0.00–0.15) ────────────────────────
    # For an AI Engineer role, active GitHub presence is a strong signal
    # of continued technical engagement. -1 means no GitHub linked.
    github = feat.get("github_activity_score", -1)
    if github is not None and github >= 0:
        if github >= 70:
            score += 0.15
        elif github >= 50:
            score += 0.12
        elif github >= 30:
            score += 0.08
        elif github >= 10:
            score += 0.04

    return max(0.0, min(1.0, score))


# ===========================================================================
# Section 7: Disqualifier Detection
# ===========================================================================
def compute_disqualifier_penalty(feat: dict) -> float:
    """
    Detects JD-specific disqualifier patterns and returns a penalty
    multiplier (0.50 – 1.0). Multiple penalties stack multiplicatively.

    Disqualifiers for Senior AI Engineer at Redrob:
      1. Title-chaser: avg tenure < 18 months across 3+ roles
      2. Entire career at consulting firms (TCS, Infosys, Wipro, etc.)
      3. Inactive > 18 months (no production code recently)
      4. offer_acceptance_rate = 0.0 (ghosts offers — red flag)
      5. interview_completion_rate < 0.5 (unreliable)
    """
    penalty = 1.0

    # ── 1. Title-chaser detection ───────────────────────────────────
    # If candidate has 3+ roles and average tenure < 18 months,
    # they're likely hopping for titles rather than building depth.
    durations = feat.get("career_durations", [])
    num_roles = feat.get("num_career_roles", 0)
    if num_roles >= 3 and durations:
        valid_durations = [d for d in durations if d > 0]
        if valid_durations:
            avg_tenure = sum(valid_durations) / len(valid_durations)
            if avg_tenure < 18:
                penalty *= 0.80  # 20% penalty for title-chasing

    # ── 2. Consulting-only career ───────────────────────────────────
    # If ALL companies in career history are consulting firms,
    # candidate likely lacks product-company / ownership experience.
    companies = feat.get("career_companies", [])
    if companies:
        consulting_count = sum(
            1 for c in companies
            if c.lower().strip() in CONSULTING_FIRMS
            or any(cf in c.lower() for cf in CONSULTING_FIRMS)
        )
        if consulting_count == len(companies):
            penalty *= 0.75  # 25% penalty for consulting-only

    # ── 3. offer_acceptance_rate = 0.0 (ghosting) ───────────────────
    oar = feat.get("offer_acceptance_rate", -1)
    if oar is not None and oar == 0.0:
        penalty *= 0.85  # 15% penalty — ghosts offers

    # ── 4. interview_completion_rate < 0.5 (unreliable) ─────────────
    icr = feat.get("interview_completion_rate", 0)
    if icr is not None and icr < 0.5 and icr >= 0:
        penalty *= 0.90  # 10% penalty — misses interviews

    # ── 5. recruiter_response_rate < 0.2 (unresponsive) ─────────────
    rr = feat.get("recruiter_response_rate", 0) or 0
    if rr < 0.2:
        penalty *= 0.92  # 8% penalty — doesn't respond to recruiters

    return max(0.50, penalty)  # Never penalize more than 50%


# ===========================================================================
# Section 8: Keyword-Stuffing / Additional Honeypot Detection
# ===========================================================================
def compute_honeypot_penalty(feat: dict) -> float:
    """
    Secondary honeypot detection via statistical anomalies.
    Returns a penalty multiplier (0.30 – 1.0).

    This catches cases that the simple is_honeypot() misses:
      1. Many skills (15+) but ALL have 0 endorsements
      2. All skills at expert/advanced level but 0 endorsements
      3. Unrealistic skill density (>6 skills per year of experience)
      4. Assessment scores contradict proficiency claims
    """
    num_skills = feat.get("num_skills", 0)
    total_endorsements = feat.get("total_endorsements", 0)
    yoe = feat.get("years_of_experience", 0) or 0
    proficiencies = feat.get("skill_proficiencies", {})
    assessments = feat.get("assessment_scores", {})

    penalty = 1.0

    if num_skills == 0:
        return 1.0

    # Check 1: Many skills, zero total endorsements
    if num_skills >= 12 and total_endorsements == 0:
        penalty *= 0.50
    elif num_skills >= 8 and total_endorsements == 0:
        penalty *= 0.70

    # Check 2: All expert/advanced but no endorsements
    advanced_expert_count = sum(
        1 for p in proficiencies.values()
        if p in ("advanced", "expert")
    )
    if advanced_expert_count >= 8 and total_endorsements < 5:
        penalty *= 0.65

    # Check 3: Unrealistic skill density
    if yoe > 0 and num_skills / yoe > 6:
        penalty *= 0.85

    # Check 4: Assessment scores contradict proficiency claims
    if assessments:
        low_assessment_count = sum(
            1 for score_val in assessments.values()
            if isinstance(score_val, (int, float)) and score_val < 25
        )
        if low_assessment_count >= 3:
            penalty *= 0.80

    return max(0.30, penalty)


# ===========================================================================
# Section 9: Required Skills Matching (for reasoning text)
# ===========================================================================
def count_required_skills_matched(feat: dict, jd_required_skills: list) -> tuple:
    """
    Counts how many JD required skills the candidate has,
    using fuzzy substring matching.

    Returns (matched_count, total_required, matched_skill_names).
    """
    if not jd_required_skills:
        return 0, 0, []

    cand_skills_lower = feat.get("skill_names_lower", [])
    headline_lower = (feat.get("headline") or "").lower()
    summary_lower = (feat.get("summary") or "").lower()

    matched_names = []
    for jd_skill in jd_required_skills:
        jd_lower = jd_skill.lower().strip()
        found = False

        # Check skill names (exact or substring match)
        for cs in cand_skills_lower:
            if jd_lower in cs or cs in jd_lower:
                found = True
                break

        # Also check headline and summary for skill mentions
        if not found:
            if jd_lower in headline_lower or jd_lower in summary_lower:
                found = True

        if found:
            matched_names.append(jd_skill)

    return len(matched_names), len(jd_required_skills), matched_names


# ===========================================================================
# Section 10: Keyword Fallback Score (when embeddings are unavailable)
# ===========================================================================
def _keyword_fallback_score(feat: dict, required_skills: list,
                            nice_to_have: list) -> float:
    """
    Simple keyword-based scoring fallback when precomputed embeddings
    are not available. Returns a score between 0.0 and 1.0.
    """
    if not required_skills:
        return 0.5

    cand_skills_lower = feat.get("skill_names_lower", [])
    headline_lower = (feat.get("headline") or "").lower()
    summary_lower = (feat.get("summary") or "").lower()

    # Check required skills
    req_matched = 0
    for skill in required_skills:
        sl = skill.lower()
        if any(sl in cs or cs in sl for cs in cand_skills_lower):
            req_matched += 1
        elif sl in headline_lower or sl in summary_lower:
            req_matched += 1

    req_score = req_matched / len(required_skills) if required_skills else 0.0

    # Check nice-to-have skills
    nth_matched = 0
    if nice_to_have:
        for skill in nice_to_have:
            sl = skill.lower()
            if any(sl in cs or cs in sl for cs in cand_skills_lower):
                nth_matched += 1
        nth_score = nth_matched / len(nice_to_have)
    else:
        nth_score = 0.0

    # Weighted blend: 80% required, 20% nice-to-have
    return 0.80 * req_score + 0.20 * nth_score


# ===========================================================================
# Section 11: Fallback Reasoning Generator
# ===========================================================================
def generate_fallback_reasoning(feat: dict, jd_data: dict,
                                skills_matched: int,
                                total_required: int,
                                matched_names: list = None) -> str:
    """
    Generates a programmatic reasoning string when the LLM-generated
    reasoning cache doesn't have an entry for this candidate.

    Format:
      "Senior AI Engineer with 7.0 yrs; 6/8 required skills matched
       (Python, FAISS, embeddings, ...); Fintech background; response rate 0.76."
    """
    title = feat.get("current_title", "Professional")
    yoe = feat.get("years_of_experience", 0)
    rr = feat.get("recruiter_response_rate", 0) or 0
    industry = feat.get("current_industry", "")
    github = feat.get("github_activity_score", -1)

    parts = [f"{title} with {yoe:.1f} yrs"]
    
    if matched_names and len(matched_names) > 0:
        # Show up to 4 matched skill names for specificity
        skill_preview = ", ".join(matched_names[:4])
        if len(matched_names) > 4:
            skill_preview += f" +{len(matched_names) - 4} more"
        parts.append(f"{skills_matched}/{total_required} required skills matched ({skill_preview})")
    else:
        parts.append(f"{skills_matched}/{total_required} required skills matched")

    if industry:
        parts.append(f"{industry} background")

    parts.append(f"response rate {rr:.2f}")

    if github is not None and github >= 0:
        parts.append(f"GitHub activity {github}")

    notice = feat.get("notice_period_days", 90)
    if notice is not None and notice > 90:
        parts.append(f"notice period {notice}d")

    return "; ".join(parts) + "."


# ===========================================================================
# Section 12: CSV Writer
# ===========================================================================
def write_submission_csv(ranked: list, output_path: str):
    """
    Writes the submission CSV with exactly the required format:
      candidate_id,rank,score,reasoning

    Rules enforced:
      - Header row + exactly 100 data rows
      - UTF-8 encoding
      - Scores are 4 decimal places
      - Non-increasing scores by rank
    """
    # Ensure output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        for cand in ranked:
            # Clean reasoning: remove newlines, limit length
            reasoning = (cand.get("reasoning") or "").replace("\n", " ").replace("\r", " ")
            if len(reasoning) > 500:
                reasoning = reasoning[:497] + "..."

            writer.writerow([
                cand["candidate_id"],
                cand["rank"],
                f"{cand['score']:.4f}",
                reasoning,
            ])


# ===========================================================================
# Section 13: File Opener Helper (supports .jsonl and .jsonl.gz)
# ===========================================================================
def _open_candidates_file(path: str):
    """
    Returns an appropriate file handle for .jsonl or .jsonl.gz files.
    Caller is responsible for closing (use with context manager).
    """
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    else:
        return open(path, "r", encoding="utf-8")


def stream_candidates(candidates_path: str):
    """
    Yields (raw_candidate_dict, is_parse_error) for each candidate.
    Supports both JSON array (.json) and JSON Lines (.jsonl/.gz) formats.
    """
    # Peek at first character to detect JSON array format
    is_json_array = False
    try:
        with _open_candidates_file(candidates_path) as f:
            for line in f:
                line_strip = line.strip()
                if line_strip:
                    if line_strip.startswith("["):
                        is_json_array = True
                    break
    except Exception:
        pass

    if is_json_array:
        with _open_candidates_file(candidates_path) as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        yield item, False
                else:
                    yield data, False
            except Exception:
                yield None, True
    else:
        with _open_candidates_file(candidates_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                    yield raw, False
                except json.JSONDecodeError:
                    yield None, True


# ===========================================================================
# Section 14: Main Ranking Pipeline
# ===========================================================================
def run_ranking(candidates_path: str, output_path: str, jd_path: str = None,
                precomputed_dir: str = None, reasoning_cache_path: str = None):
    """
    Main entry point. Loads all data, scores 100K candidates, outputs CSV.

    Architecture:
      1. Load pre-parsed JD and precomputed embeddings (both offline-generated)
      2. Stream candidates from JSONL/JSONL.GZ
      3. For each candidate: extract features → check honeypot → compute 4 scores
      4. Blend scores with weights → apply penalties
      5. Sort, assign ranks, write CSV

    Runtime target: < 90 seconds for 100K candidates.
    Memory target: < 1 GB peak.
    """
    t_start = time.time()

    # ── Resolve paths ───────────────────────────────────────────────
    if jd_path is None:
        jd_path = os.path.join(BASE_DIR, "data", "parsed_jd.json")
    if precomputed_dir is None:
        precomputed_dir = os.path.join(BASE_DIR, "precomputed")
    if reasoning_cache_path is None:
        reasoning_cache_path = os.path.join(BASE_DIR, "data", "reasoning_cache.json")

    # ── 1. Load JD ──────────────────────────────────────────────────
    print(f"[1/6] Loading parsed JD from {jd_path}...")
    with open(jd_path, "r", encoding="utf-8") as f:
        jd_data = json.load(f)

    jd_seniority = jd_data.get("seniority", "Senior")
    jd_required_skills = jd_data.get("required_skills", [])
    jd_nice_to_have = jd_data.get("nice_to_have", [])
    print(f"    JD: {jd_seniority} level, {len(jd_required_skills)} required skills, "
          f"{len(jd_nice_to_have)} nice-to-have")

    # ── 2. Load precomputed embeddings ──────────────────────────────
    emb_path = os.path.join(precomputed_dir, "embeddings.npy")
    
    # Dynamically select which JD embedding file to load based on the JD filename
    jd_emb_filename = "jd_embedding.npy"
    if jd_path and "parsed_jd_2" in jd_path:
        test_path = os.path.join(precomputed_dir, "jd_embedding_2.npy")
        if os.path.exists(test_path):
            jd_emb_filename = "jd_embedding_2.npy"
            print(f"    Using precomputed PM JD embedding: {jd_emb_filename}")
        else:
            print(f"    WARNING: precomputed/jd_embedding_2.npy not found! Fallback to default.")
            
    jd_emb_path = os.path.join(precomputed_dir, jd_emb_filename)
    ids_path = os.path.join(precomputed_dir, "candidate_ids.json")

    has_embeddings = all(os.path.exists(p) for p in [emb_path, jd_emb_path, ids_path])

    if has_embeddings:
        print(f"[2/6] Loading precomputed embeddings from {precomputed_dir}...")
        embeddings = np.load(emb_path)
        jd_embedding = np.load(jd_emb_path)
        with open(ids_path, "r") as f:
            precomputed_ids = json.load(f)

        # Build id → index mapping for O(1) lookup
        id_to_idx = {cid: idx for idx, cid in enumerate(precomputed_ids)}

        # Vectorized cosine similarity for ALL candidates at once
        print(f"    Computing cosine similarities for {len(precomputed_ids)} embeddings...")
        # Convert to float32 for computation if stored as float16
        if embeddings.dtype == np.float16:
            embeddings = embeddings.astype(np.float32)

        emb_norms = np.linalg.norm(embeddings, axis=1)
        jd_norm = np.linalg.norm(jd_embedding)
        emb_norms[emb_norms == 0] = 1.0
        if jd_norm == 0:
            jd_norm = 1.0
        all_semantic_scores = embeddings.dot(jd_embedding) / (emb_norms * jd_norm)
        print(f"    Semantic scores computed. Range: [{all_semantic_scores.min():.4f}, "
              f"{all_semantic_scores.max():.4f}]")
    else:
        print(f"[2/6] WARNING: Precomputed embeddings not found at {precomputed_dir}")
        print(f"    Running without semantic scores. Run precompute.py first for best results.")
        id_to_idx = {}
        all_semantic_scores = None

    # ── 3. Load reasoning cache ─────────────────────────────────────
    if os.path.exists(reasoning_cache_path):
        print(f"[3/6] Loading reasoning cache from {reasoning_cache_path}...")
        with open(reasoning_cache_path, "r", encoding="utf-8") as f:
            reasoning_cache = json.load(f)
        print(f"    Loaded {len(reasoning_cache)} cached reasoning entries.")
    else:
        print(f"[3/6] No reasoning cache found. Will generate fallback reasoning.")
        reasoning_cache = {}

    # ── 4. Stream candidates and score ──────────────────────────────
    print(f"[4/6] Streaming candidates from {candidates_path}...")
    today = date.today()
    scored_candidates = []
    line_count = 0
    errors = 0
    honeypots_detected = 0

    for raw, is_error in stream_candidates(candidates_path):
        if is_error:
            errors += 1
            continue

        candidate_id = raw.get("candidate_id") or raw.get("id") or ""

        # ── Honeypot check (BEFORE any scoring) ─────────────────
        if is_honeypot(raw):
            honeypots_detected += 1
            # Cap score at 0.05 so honeypots never reach top 100
            scored_candidates.append({
                "candidate_id": candidate_id,
                "score": 0.0500,
                "reasoning": "Profile flagged: skill proficiency claims inconsistent with duration data.",
            })
            line_count += 1
            if line_count % 10000 == 0:
                elapsed = time.time() - t_start
                print(f"    Processed {line_count:,} candidates... ({elapsed:.1f}s)")
            continue

        # ── Extract features ────────────────────────────────────
        feat = extract_candidate_features(raw)

        # ── Semantic score from precomputed embeddings ──────────
        if all_semantic_scores is not None and candidate_id in id_to_idx:
            semantic_score = float(all_semantic_scores[id_to_idx[candidate_id]])
        else:
            # Fallback: keyword-match heuristic
            semantic_score = _keyword_fallback_score(feat, jd_required_skills, jd_nice_to_have)

        # ── Seniority penalty on semantic score ─────────────────
        seniority_mult = compute_seniority_penalty(jd_seniority, feat["current_title"])

        # ── India platform signal score ─────────────────────────
        india_score = compute_india_platform_score(feat, today)

        # ── Trajectory score ────────────────────────────────────
        trajectory_score = compute_trajectory_score(feat)

        # ── Behavioral signal score ─────────────────────────────
        behavioral_score = compute_behavioral_signal_score(feat, today)

        # ── Disqualifier penalty ────────────────────────────────
        disqualifier_mult = compute_disqualifier_penalty(feat)

        # ── Secondary honeypot/stuffer penalty ──────────────────
        stuffer_mult = compute_honeypot_penalty(feat)

        # ── Required skills matched (for reasoning) ─────────────
        skills_matched, total_req, matched_names = count_required_skills_matched(
            feat, jd_required_skills
        )

        # ── Final score blending ────────────────────────────────
        # Formula: weighted sum × disqualifier × stuffer penalties
        # Seniority penalty only applies to semantic component
        final_score = disqualifier_mult * stuffer_mult * (
            W_SEMANTIC * semantic_score * seniority_mult
            + W_INDIA * india_score
            + W_TRAJECTORY * trajectory_score
            + W_BEHAVIORAL * behavioral_score
        )
        final_score = max(0.0, min(1.0, final_score))

        # ── Look up or generate reasoning ──────────────────────
        reasoning = reasoning_cache.get(candidate_id)
        if not reasoning:
            reasoning = generate_fallback_reasoning(
                feat, jd_data, skills_matched, total_req, matched_names
            )

        scored_candidates.append({
            "candidate_id": candidate_id,
            "score": round(final_score, 4),
            "reasoning": reasoning,
        })

        line_count += 1
        if line_count % 10000 == 0:
            elapsed = time.time() - t_start
            print(f"    Processed {line_count:,} candidates... ({elapsed:.1f}s)")

    print(f"    Total: {line_count:,} candidates scored, {errors} parse errors, "
          f"{honeypots_detected} honeypots detected.")

    # ── 5. Sort and assign ranks ────────────────────────────────────
    print(f"[5/6] Sorting and assigning ranks...")

    # Primary sort: score descending. Tie-break: candidate_id ascending
    scored_candidates.sort(key=lambda x: (-x["score"], x["candidate_id"]))

    # Take top 100
    top_100 = scored_candidates[:100]

    # Assign ranks 1–100
    for rank, cand in enumerate(top_100, start=1):
        cand["rank"] = rank

    # ── Verify no honeypots in top 100 ──────────────────────────────
    honeypots_in_top = sum(1 for c in top_100 if c["score"] <= 0.0500)
    if honeypots_in_top > 0:
        print(f"    WARNING: {honeypots_in_top} honeypot-scored candidates in top 100!")

    # ── 6. Write CSV ────────────────────────────────────────────────
    print(f"[6/6] Writing submission CSV to {output_path}...")
    write_submission_csv(top_100, output_path)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  RANKING COMPLETE")
    print(f"  Candidates scored: {line_count:,}")
    print(f"  Honeypots detected: {honeypots_detected}")
    print(f"  Top 100 written to: {output_path}")
    if len(top_100) > 0:
        print(f"  Score range: [{top_100[-1]['score']:.4f}, {top_100[0]['score']:.4f}]")
    else:
        print("  Score range: N/A (no candidates scored)")
    print(f"  Wall-clock time: {elapsed:.2f} seconds")
    print(f"{'='*60}")

    # Print top 5 for quick inspection
    print(f"\n  Top 5 candidates:")
    for c in top_100[:5]:
        print(f"    Rank {c['rank']:>3}: {c['candidate_id']} | "
              f"Score: {c['score']:.4f} | {c['reasoning'][:80]}...")

    return top_100


# ===========================================================================
# Section 15: CLI Entry Point
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="TalentLens Bharat — Candidate Ranking Script (zero-network, < 5 min)"
    )
    parser.add_argument(
        "--candidates", "-c",
        default=os.path.join(BASE_DIR, "candidates.jsonl"),
        help="Path to candidates JSONL or JSONL.GZ file (default: ./candidates.jsonl)"
    )
    parser.add_argument(
        "--out", "-o",
        default=os.path.join(BASE_DIR, "submission.csv"),
        help="Output CSV path (default: ./submission.csv)"
    )
    parser.add_argument(
        "--jd",
        default=None,
        help="Path to parsed JD JSON (default: data/parsed_jd.json)"
    )
    parser.add_argument(
        "--precomputed-dir",
        default=None,
        help="Path to precomputed embeddings directory (default: precomputed/)"
    )
    parser.add_argument(
        "--reasoning-cache",
        default=None,
        help="Path to reasoning cache JSON (default: data/reasoning_cache.json)"
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.candidates):
        print(f"ERROR: Candidates file not found: {args.candidates}", file=sys.stderr)
        sys.exit(1)

    run_ranking(
        candidates_path=args.candidates,
        output_path=args.out,
        jd_path=args.jd,
        precomputed_dir=args.precomputed_dir,
        reasoning_cache_path=args.reasoning_cache,
    )


if __name__ == "__main__":
    main()
