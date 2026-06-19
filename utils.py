import os
import json
import logging
from typing import Any, Dict
from dotenv import load_dotenv

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Configures standard logging with a clean, descriptive format.
    In production, this could be extended to output JSON or log to centralized monitoring.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger("talentlens")

logger = setup_logging()

def load_environment() -> None:
    """
    Loads environment variables from a .env file if it exists,
    providing warnings if critical keys are missing.
    """
    load_dotenv()
    if not os.environ.get("GROQ_API_KEY"):
        logger.warning(
            "GROQ_API_KEY environment variable is not set! "
            "Please check your .env file or export it directly in your terminal shell."
        )
    else:
        logger.info("Successfully checked/loaded environment variables.")

def ensure_directories_exist(paths: list[str]) -> None:
    """
    Helper function to ensure all required directories exist.
    """
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            logger.info(f"Created directory: {path}")

def save_json(data: Any, filepath: str) -> None:
    """
    Utility to write data to JSON with pretty print and proper encoding.
    Ensures directory structure is created before writing.
    """
    directory = os.path.dirname(filepath)
    if directory:
        ensure_directories_exist([directory])
        
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Successfully saved data to: {filepath}")
    except IOError as e:
        logger.error(f"Failed to write JSON data to {filepath}: {str(e)}")
        raise

def load_json(filepath: str) -> Any:
    """
    Utility to load and deserialize a JSON file with error handling.
    """
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"JSON file not found: {filepath}")
        
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded data from: {filepath}")
        return data
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Failed to parse JSON file {filepath}: {str(e)}")
        raise

def compute_confidence_reasoning(cand: dict, jd_data: dict) -> str:
    """
    Computes a structured confidence score with reasoning.
    Example output: "High confidence (0.77) — 4/5 required skills matched, seniority exact, active 3 days ago."
    """
    from stage2_ranker import calculate_seniority_penalty
    
    score = cand.get("final_score", cand.get("match_score", 0.0))
    
    # Confidence Level
    if score >= 0.72:
        level = "High confidence"
    elif score >= 0.60:
        level = "Medium confidence"
    else:
        level = "Low confidence"
        
    # Required skills matched
    req_skills = jd_data.get("required_skills", [])
    cand_skills = cand.get("skills") or []
    cand_skills_lower = [s.lower().strip() for s in cand_skills]
    
    matched_count = 0
    for rs in req_skills:
        rs_lower = rs.lower().strip()
        if any(rs_lower in cs or cs in rs_lower for cs in cand_skills_lower):
            matched_count += 1
            
    skills_text = f"{matched_count}/{len(req_skills)} required skills matched" if req_skills else "0/0 required skills matched"
    
    # Seniority Match
    jd_seniority = jd_data.get("seniority", "Mid")
    current_title = cand.get("current_title", "")
    _, seniority_match = calculate_seniority_penalty(jd_seniority, current_title)
    seniority_text = "seniority exact" if seniority_match else "seniority mismatch"
    
    # Activity
    active_text = "activity unknown"
    for sig in cand.get("india_signals_detected", []):
        if "active" in sig.lower():
            active_text = sig.replace("Highly active: ", "").replace("Recently active: ", "").replace("Inactive penalty: ", "")
            break
            
    return f"{level} ({score:.2f}) — {skills_text}, {seniority_text}, {active_text}."

def select_dark_horse(candidates: list, limit: int = 5) -> dict:
    """
    Heuristically selects one candidate outside the top N (limit) with high potential.
    Returns a dictionary with candidate details and reasoning.
    """
    eligible = candidates[limit:]
    if not eligible:
        return None
        
    best_candidate = None
    best_reason = ""
    best_pot = -999.0
    
    for cand in eligible:
        score_multiplier = 0.0
        reasons = []
        
        trajectory = cand.get("trajectory_label", "")
        if trajectory == "High momentum":
            score_multiplier += 0.4
            reasons.append("high-velocity career momentum (acquiring senior-level skills rapidly)")
        elif trajectory == "Steady growth":
            score_multiplier += 0.2
            reasons.append("steady upward skill trajectory")
            
        location_signals = [sig for sig in cand.get("india_signals_detected", []) if "Tier-2/3" in sig]
        if location_signals:
            score_multiplier += 0.3
            reasons.append(f"strong representation from a Tier-2/3 city ({location_signals[0].split(': ')[-1]})")
            
        recent_skills = cand.get("recent_skills_added", [])
        if recent_skills:
            score_multiplier += 0.2
            reasons.append(f"recently added highly relevant skills ({', '.join(recent_skills[:2])})")
            
        if cand.get("recruiter_flag") and "Inactive" in cand.get("recruiter_flag"):
            score_multiplier -= 0.3
            
        # Combine with their baseline semantic score
        dark_horse_potential = cand.get("semantic_score", 0.0) + score_multiplier
        
        if dark_horse_potential > best_pot:
            best_pot = dark_horse_potential
            best_candidate = cand
            if reasons:
                best_reason = f"{cand.get('name')} shows great potential due to: " + ", ".join(reasons) + "."
            else:
                best_reason = f"{cand.get('name')} shows promising core semantic alignment despite a lower seniority or signal adjustments."
                
    if best_candidate:
        return {
            "id": best_candidate.get("id"),
            "name": best_candidate.get("name"),
            "current_title": best_candidate.get("current_title"),
            "rank": best_candidate.get("rank"),
            "final_score": best_candidate.get("final_score"),
            "reason": best_reason
        }
    return None

