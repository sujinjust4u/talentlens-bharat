import os
import sys
import numpy as np

# Add current directory to path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import utils

logger = utils.setup_logging()

# =====================================================================
# 1. Text Representation Construction (Profile Building)
# =====================================================================
def build_jd_profile_text(jd: dict) -> str:
    """
    Transforms the structured JSON output of Stage 1 into a coherent textual document.
    By structuring the text with clear headings, we help the embedding model capture
    semantic correlations between job requirements and candidate skills.
    """
    parts = []
    
    # 1. Title/Seniority and Domain context
    seniority = jd.get("seniority", "Mid-Level")
    domain = jd.get("domain", "General Technology")
    parts.append(f"Looking for a {seniority} level professional working in the {domain} field.")
    
    # 2. Required core competencies
    req_skills = jd.get("required_skills", [])
    if req_skills:
        parts.append(f"Required core skills and technologies: {', '.join(req_skills)}.")
        
    # 3. Non-mandatory preferences
    nice_to_have = jd.get("nice_to_have", [])
    if nice_to_have:
        parts.append(f"Nice to have preferences: {', '.join(nice_to_have)}.")
        
    # 4. Inferred requirements (workplace style, scaling, culture)
    implicit = jd.get("implicit_signals", [])
    if implicit:
        parts.append(f"Contextual requirements: {', '.join(implicit)}.")
        
    return " ".join(parts)


def build_candidate_profile_text(cand: dict) -> str:
    """
    Transforms a candidate JSON record into a coherent textual document.
    Gracefully handles missing or null fields by checking type and contents,
    preventing runtime exceptions during batch processing.
    """
    parts = []
    
    # 1. Current position & domain
    title = cand.get("current_title")
    domain = cand.get("domain")
    if title and domain:
        parts.append(f"Currently working as a {title} in the {domain} industry.")
    elif title:
        parts.append(f"Currently working as a {title}.")
        
    # 2. Years of Experience
    exp = cand.get("experience_years")
    if exp is not None:
        parts.append(f"Has {exp} years of professional experience.")
        
    # 3. Technical Skills
    skills = cand.get("skills")
    if isinstance(skills, list) and skills:
        parts.append(f"Skills and technologies: {', '.join(skills)}.")
        
    # 4. Previous Career History
    prev_titles = cand.get("previous_titles")
    if isinstance(prev_titles, list) and prev_titles:
        parts.append(f"Previous titles: {', '.join(prev_titles)}.")
        
    # 5. Biography / Summary
    bio = cand.get("bio")
    if bio:
        parts.append(f"Summary: {bio}")
        
    return " ".join(parts)


# =====================================================================
# 2. Seniority Levels and Match Penalty
# =====================================================================
SENIORITY_LEVELS = {
    "intern": 0,
    "junior": 1,
    "mid": 2,
    "senior": 3,
    "lead": 4,
    "principal": 5,
    "architect": 5,
    "staff": 5
}

def map_seniority_to_level(text: str) -> int:
    """
    Infers the seniority level integer (0-5) from a title or description.
    Defaults to 2 (mid-level) if title is missing, null, or unspecified.
    """
    if not text:
        return 2  # default mid-level assumption
    
    text_lower = text.lower()
    if "principal" in text_lower or "architect" in text_lower or "staff" in text_lower:
        return 5
    elif "lead" in text_lower:
        return 4
    elif "senior" in text_lower:
        return 3
    elif "junior" in text_lower:
        return 1
    elif "intern" in text_lower:
        return 0
    elif "mid" in text_lower or "middle" in text_lower or "associate" in text_lower:
        return 2
        
    return 2  # Default Mid assumption for unmapped roles


def calculate_seniority_penalty(jd_seniority: str, candidate_title: str) -> tuple[float, bool]:
    """
    Compares the JD seniority level with the candidate's current title seniority.
    Returns:
        - multiplier: float representing the score penalty.
        - seniority_match: bool indicating if the gap is 0.
    """
    jd_level = map_seniority_to_level(jd_seniority)
    cand_level = map_seniority_to_level(candidate_title)
    
    gap = abs(jd_level - cand_level)
    
    if gap == 0:
        return 1.0, True
    elif gap == 1:
        return 0.92, False
    elif gap == 2:
        return 0.82, False
    else:
        return 0.75, False


# =====================================================================
# 3. Embedding & Cosine Similarity Ranking
# =====================================================================
def rank_candidates(jd_file: str, candidates_file: str, output_file: str) -> list:
    """
    Main orchestration function for Stage 2.
    Embeds the Job Description and all Candidates, calculates similarity,
    applies a seniority match penalty, ranks them, and writes the output.
    """
    # 1. Load data
    logger.info("Loading inputs for Semantic Ranker...")
    jd_data = utils.load_json(jd_file)
    candidates_list = utils.load_json(candidates_file)
    
    if not candidates_list:
        logger.error("Candidate list is empty.")
        raise ValueError("Candidates list cannot be empty.")
        
    # 2. Initialize Sentence Transformer Model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    logger.info(f"Loading local SentenceTransformer model: {model_name}...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
    except Exception as e:
        logger.error(f"Failed to load sentence-transformers library: {str(e)}")
        raise e
        
    # 3. Build text documents for embeddings
    jd_text = build_jd_profile_text(jd_data)
    candidate_texts = []
    for cand in candidates_list:
        text = build_candidate_profile_text(cand)
        candidate_texts.append(text)
        logger.info(f"Built profile text for candidate {cand.get('id', 'Unknown')}: {text[:100]}...")
        
    # 4. Compute embeddings
    logger.info(f"Embedding {len(candidate_texts)} candidate profiles + 1 Job Description...")
    jd_embedding = model.encode(jd_text, convert_to_numpy=True)
    cand_embeddings = model.encode(candidate_texts, convert_to_numpy=True)
    
    # 5. Calculate Cosine Similarities in batch using NumPy
    jd_norm = np.linalg.norm(jd_embedding)
    cand_norms = np.linalg.norm(cand_embeddings, axis=1)
    
    if jd_norm == 0:
        jd_norm = 1.0
    cand_norms[cand_norms == 0] = 1.0
    
    dot_products = np.dot(cand_embeddings, jd_embedding)
    cosine_scores = dot_products / (cand_norms * jd_norm)
    
    # 6. Construct output list with scores, penalties and ranks
    scored_candidates = []
    jd_seniority = jd_data.get("seniority", "")
    
    for idx, cand in enumerate(candidates_list):
        scored_cand = cand.copy()
        
        # Calculate raw semantic match
        semantic_score = float(cosine_scores[idx])
        
        # Calculate seniority penalty
        current_title = cand.get("current_title", "")
        multiplier, seniority_match = calculate_seniority_penalty(jd_seniority, current_title)
        final_score = semantic_score * multiplier
        
        # Store metadata
        scored_cand["semantic_score"] = round(semantic_score, 4)
        scored_cand["final_score"] = round(final_score, 4)
        scored_cand["seniority_match"] = seniority_match
        
        # Set match_score = final_score for orchestrator compatibility
        scored_cand["match_score"] = scored_cand["final_score"]
        
        scored_candidates.append(scored_cand)
        
    # Sort candidates by final_score descending
    scored_candidates.sort(key=lambda x: x["final_score"], reverse=True)
    
    # Assign ranks
    for rank_idx, cand in enumerate(scored_candidates):
        cand["rank"] = rank_idx + 1
        
    # 7. Save results
    utils.save_json(scored_candidates, output_file)
    logger.info(f"Stage 2 complete. Ranked candidates saved to: {output_file}")
    
    return scored_candidates


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    jd_json = os.path.join(base_dir, "data", "parsed_jd.json")
    candidates_json = os.path.join(base_dir, "data", "candidates.json")
    output_json = os.path.join(base_dir, "output", "ranked_candidates.json")
    
    try:
        ranked = rank_candidates(jd_json, candidates_json, output_json)
        
        print("\n=== SUCCESS: TOP CANDIDATES RANKING ===")
        for cand in ranked[:4]:
            print(f"Rank {cand['rank']}: {cand['name']} (ID: {cand['id']})")
            print(f"  Title: {cand.get('current_title')} | Seniority Match: {cand.get('seniority_match')}")
            print(f"  Scores -> Semantic Cosine: {cand['semantic_score']} | Final Penalty-Adjusted: {cand['final_score']}")
        print("=======================================\n")
        
    except Exception as e:
        print(f"\n[ERROR] Stage 2 failed to execute: {str(e)}", file=sys.stderr)
        sys.exit(1)
