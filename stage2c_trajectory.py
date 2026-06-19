"""
TalentLens Stage 2c: Career Trajectory Predictor

Predicts career trajectory for each candidate based on skill velocity,
title progression speed, and advanced recency signals. Enriches candidate
records with trajectory_score, trajectory_label, and trajectory_signals.

Supports both old (flat) and real (nested) candidate schemas.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import utils

logger = utils.setup_logging()

# =====================================================================
# 1. Skills Seniority Map
# =====================================================================
SKILLS_BY_SENIORITY = {
    "junior": {
        "git", "html", "css", "basic python", "mysql", "rest apis",
        "sql", "excel", "jira", "basic sql", "javascript", "bootstrap",
    },
    "mid": {
        "docker", "postgresql", "redis", "fastapi", "django", "aws basics",
        "ci/cd", "flask", "aws", "sqlalchemy", "pytest", "celery",
        "mongodb", "elasticsearch", "graphql", "typescript", "react",
    },
    "senior": {
        "kubernetes", "langchain", "llms", "system design", "vector databases",
        "kafka", "terraform", "microservices architecture", "microservices", "grpc",
        "hugging face", "huggingface", "pytorch", "scikit-learn", "sklearn",
        "airflow", "deep learning", "nlp", "natural language processing",
        "computer vision", "embeddings", "sentence-transformers",
        "transformers", "transformer", "prompt engineering",
    },
    "lead": {
        "ml ops", "mlops", "fine tuning", "fine-tuning", "rag pipelines", "rag",
        "distributed systems", "technical roadmap", "platform architecture",
        "mlflow", "model serving", "feature stores", "kubeflow", "ray",
        "lora", "qlora", "peft", "learning-to-rank",
        "faiss", "pinecone", "qdrant", "milvus", "weaviate", "chroma",
    }
}

SENIORITY_RANK = {"junior": 0, "mid": 1, "senior": 2, "lead": 3}


# =====================================================================
# 2. Schema Adaptation Helpers
# =====================================================================
def _get_current_title(cand: dict) -> str:
    """Extracts current_title from either flat or nested schema."""
    title = cand.get("current_title", "")
    if title:
        return title
    profile = cand.get("profile") or {}
    return profile.get("current_title", "")


def _get_skills_as_strings(cand: dict) -> list:
    """
    Returns a flat list of skill name strings from either:
      - Old schema: cand["skills"] = ["Python", "FastAPI"]
      - Real schema: cand["skills"] = [{"name": "Python", "proficiency": "advanced", ...}]
    """
    skills = cand.get("skills")
    if not skills or not isinstance(skills, list):
        return []
    
    result = []
    for s in skills:
        if isinstance(s, dict):
            name = (s.get("name") or "").strip()
            if name:
                result.append(name)
        elif isinstance(s, str):
            result.append(s.strip())
    return result


def _get_previous_titles(cand: dict) -> list:
    """
    Returns list of previous title strings from either:
      - Old schema: cand["previous_titles"] = ["Junior Dev", "Intern"]
      - Real schema: cand["career_history"] = [{title, is_current, ...}]
    """
    # Try old schema first
    prev = cand.get("previous_titles")
    if isinstance(prev, list) and prev:
        return prev
    
    # Try real schema
    career = cand.get("career_history") or []
    titles = []
    for role in career:
        if isinstance(role, dict) and not role.get("is_current"):
            title = role.get("title", "")
            if title:
                titles.append(title)
    return titles


def _get_experience_years(cand: dict) -> float:
    """Returns years of experience from either flat or nested schema."""
    yoe = cand.get("experience_years")
    if yoe is not None:
        return float(yoe)
    profile = cand.get("profile") or {}
    yoe = profile.get("years_of_experience", 0)
    return float(yoe) if yoe else 0.0


def _get_recent_skills(cand: dict) -> list:
    """
    Returns recently-added skills from either:
      - Old schema: cand["recent_skills_added"] = ["LangChain", "Terraform"]
      - Real schema: skills with duration_months < 12
    """
    # Try old schema first
    recent = cand.get("recent_skills_added")
    if isinstance(recent, list) and recent:
        return recent
    
    # Real schema: find skills with short duration
    skills = cand.get("skills") or []
    recent_skills = []
    for s in skills:
        if isinstance(s, dict):
            duration = s.get("duration_months", 999)
            name = (s.get("name") or "").strip()
            if name and isinstance(duration, (int, float)) and 0 < duration < 12:
                recent_skills.append(name)
    return recent_skills


def _get_candidate_name(cand: dict) -> str:
    """Returns candidate name from either flat or nested schema."""
    name = cand.get("name", "")
    if name:
        return name
    profile = cand.get("profile") or {}
    return profile.get("anonymized_name", "Unknown")


def _infer_seniority_from_title(title: str) -> str:
    """Maps a candidate's current title to a seniority bucket."""
    if not title:
        return "mid"
    t = title.lower()
    if "lead" in t or "principal" in t or "architect" in t or "staff" in t:
        return "lead"
    elif "senior" in t or "sr." in t or "sr " in t:
        return "senior"
    elif "junior" in t or "intern" in t or "trainee" in t:
        return "junior"
    return "mid"


def _count_skills_at_level(skills: list, level: str) -> list:
    """Returns list of candidate skills that belong to a given seniority level."""
    if not skills or not isinstance(skills, list):
        return []
    level_skills = SKILLS_BY_SENIORITY.get(level, set())
    return [s for s in skills if s.strip().lower() in level_skills]


# =====================================================================
# 3. Core Trajectory Scorer
# =====================================================================
def compute_trajectory_score(candidate: dict, jd_seniority: str = "Mid") -> dict:
    """
    Computes career trajectory signals for a candidate.
    
    Supports both old (flat) and real (nested) candidate schemas.

    Returns a dict with:
      - trajectory_score (float, 0.0 to 1.0)
      - trajectory_label (str)
      - trajectory_signals (list of str)
    """
    signals = []
    skill_velocity_score = 0.0
    title_speed_score = 0.0
    advanced_recency_score = 0.0

    current_title = _get_current_title(candidate)
    cand_seniority = _infer_seniority_from_title(current_title)
    cand_rank = SENIORITY_RANK.get(cand_seniority, 1)

    skills = _get_skills_as_strings(candidate)
    previous_titles = _get_previous_titles(candidate)
    experience_years = _get_experience_years(candidate)
    recent_skills = _get_recent_skills(candidate)

    # -----------------------------------------------------------------
    # 3.1 Skill Velocity (0.0 – 0.4)
    # Count skills belonging to levels ABOVE the candidate's current title
    # -----------------------------------------------------------------
    above_level_skills = []
    for level_name, level_rank in SENIORITY_RANK.items():
        if level_rank > cand_rank:
            matched = _count_skills_at_level(skills, level_name)
            above_level_skills.extend(matched)

    if len(above_level_skills) >= 4:
        skill_velocity_score = 0.4
    elif len(above_level_skills) >= 3:
        skill_velocity_score = 0.3
    elif len(above_level_skills) >= 2:
        skill_velocity_score = 0.2
    elif len(above_level_skills) >= 1:
        skill_velocity_score = 0.1

    if above_level_skills:
        signals.append(
            f"{len(above_level_skills)} higher-level skill(s) detected above current title: "
            f"{', '.join(above_level_skills[:5])}"
        )

    # -----------------------------------------------------------------
    # 3.2 Title Progression Speed (0.0 – 0.3)
    # Fast movers: 2+ previous titles with < 5 years experience
    # -----------------------------------------------------------------
    num_prev_titles = len(previous_titles)
    total_titles = num_prev_titles + 1  # include current title

    if num_prev_titles >= 2 and experience_years < 5:
        title_speed_score = 0.3
        signals.append(
            f"Fast title progression: {total_titles} titles in {experience_years:.0f} years"
        )
    elif num_prev_titles >= 2 and experience_years < 7:
        title_speed_score = 0.2
        signals.append(
            f"Solid title progression: {total_titles} titles in {experience_years:.0f} years"
        )
    elif num_prev_titles >= 1:
        title_speed_score = 0.1
        signals.append(
            f"Standard progression: {total_titles} titles in {experience_years:.0f} years"
        )

    # -----------------------------------------------------------------
    # 3.3 Advanced Recency (0.0 – 0.3)
    # If recent_skills contains senior or lead level skills
    # -----------------------------------------------------------------
    if isinstance(recent_skills, list) and recent_skills:
        advanced_recent = []
        for level in ["senior", "lead"]:
            matched = _count_skills_at_level(recent_skills, level)
            advanced_recent.extend(matched)

        if len(advanced_recent) >= 3:
            advanced_recency_score = 0.3
        elif len(advanced_recent) >= 2:
            advanced_recency_score = 0.2
        elif len(advanced_recent) >= 1:
            advanced_recency_score = 0.1

        if advanced_recent:
            signals.append(
                f"Recently added {len(advanced_recent)} advanced skill(s): "
                f"{', '.join(advanced_recent[:5])}"
            )

    # -----------------------------------------------------------------
    # 3.4 Aggregate and Label
    # -----------------------------------------------------------------
    raw_score = skill_velocity_score + title_speed_score + advanced_recency_score
    trajectory_score = max(0.0, min(1.0, raw_score))

    if trajectory_score >= 0.7:
        trajectory_label = "High momentum"
    elif trajectory_score >= 0.5:
        trajectory_label = "Steady growth"
    elif trajectory_score >= 0.3:
        trajectory_label = "Early stage"
    else:
        trajectory_label = "Plateau"

    return {
        "trajectory_score": round(trajectory_score, 4),
        "trajectory_label": trajectory_label,
        "trajectory_signals": signals
    }


# =====================================================================
# 4. Batch Enrichment Function
# =====================================================================
def apply_trajectory(ranked_candidates: list, jd_seniority: str = "Mid") -> list:
    """
    Enriches each candidate with trajectory fields and recalculates final_score
    using the 70/20/10 formula:
        final_score = 0.70 × semantic_score + 0.20 × india_signal_score + 0.10 × trajectory_score

    Re-sorts candidates by final_score descending and re-assigns ranks.
    
    Works with both old (flat) and real (nested) candidate schemas.
    """
    logger.info(f"Applying trajectory scoring to {len(ranked_candidates)} candidates...")

    enriched = []
    for cand in ranked_candidates:
        enriched_cand = cand.copy()

        traj = compute_trajectory_score(enriched_cand, jd_seniority)
        enriched_cand["trajectory_score"] = traj["trajectory_score"]
        enriched_cand["trajectory_label"] = traj["trajectory_label"]
        enriched_cand["trajectory_signals"] = traj["trajectory_signals"]

        # Recalculate final_score with new 70/20/10 blend
        semantic = enriched_cand.get("semantic_score", 0.0)
        india = enriched_cand.get("india_signal_score", 0.50)
        trajectory = traj["trajectory_score"]

        new_final = 0.70 * semantic + 0.20 * india + 0.10 * trajectory
        enriched_cand["final_score"] = round(new_final, 4)
        enriched_cand["match_score"] = enriched_cand["final_score"]

        cand_name = _get_candidate_name(enriched_cand)
        logger.info(
            f"Candidate {cand_name} - Trajectory: {traj['trajectory_score']:.2f} "
            f"({traj['trajectory_label']}) | New Final: {new_final:.4f}"
        )
        enriched.append(enriched_cand)

    # Re-sort by final_score descending
    enriched.sort(key=lambda x: x["final_score"], reverse=True)

    # Re-assign ranks
    for idx, cand in enumerate(enriched):
        cand["rank"] = idx + 1

    logger.info("Trajectory scoring completed.")
    return enriched


if __name__ == "__main__":
    import json

    # Test with old (flat) schema
    mock_old = [
        {
            "name": "Test Mid with Senior Skills",
            "current_title": "Backend Engineer",
            "skills": ["Python", "Docker", "Kubernetes", "LangChain", "Kafka"],
            "previous_titles": ["Junior Developer", "Intern"],
            "experience_years": 3,
            "recent_skills_added": ["LangChain", "Terraform"],
            "semantic_score": 0.70,
            "india_signal_score": 0.65
        },
    ]
    
    # Test with real (nested) schema
    mock_real = [
        {
            "candidate_id": "CAND_0000001",
            "profile": {
                "anonymized_name": "Test Real Schema",
                "current_title": "Backend Engineer",
                "years_of_experience": 4,
            },
            "skills": [
                {"name": "Python", "proficiency": "advanced", "endorsements": 10, "duration_months": 36},
                {"name": "Docker", "proficiency": "intermediate", "endorsements": 5, "duration_months": 24},
                {"name": "Kubernetes", "proficiency": "intermediate", "endorsements": 3, "duration_months": 8},
                {"name": "LangChain", "proficiency": "beginner", "endorsements": 1, "duration_months": 3},
                {"name": "Kafka", "proficiency": "intermediate", "endorsements": 2, "duration_months": 6},
            ],
            "career_history": [
                {"title": "Backend Engineer", "is_current": True, "duration_months": 18},
                {"title": "Junior Developer", "is_current": False, "duration_months": 12},
                {"title": "Intern", "is_current": False, "duration_months": 6},
            ],
            "semantic_score": 0.70,
            "india_signal_score": 0.65,
        },
    ]

    print("\n=== TEST 1: OLD (FLAT) SCHEMA ===")
    result1 = apply_trajectory(mock_old, "Senior")
    print(json.dumps(result1, indent=2))
    
    print("\n=== TEST 2: REAL (NESTED) SCHEMA ===")
    result2 = apply_trajectory(mock_real, "Senior")
    print(json.dumps(result2, indent=2))
    
    print("================================\n")
