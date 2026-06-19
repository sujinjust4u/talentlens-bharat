import os
import sys
from typing import List
from pydantic import BaseModel, Field

# Add current directory to path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import utils

logger = utils.setup_logging()

# =====================================================================
# 1. Structured Output Schemas
# =====================================================================
class SuggestedResource(BaseModel):
    name: str = Field(description="Name of the course, tutorial, or repository. No special characters.")
    url_or_platform: str = Field(description="Platform or domain providing the resource, e.g. Coursera, GitHub, or Official Docs.")
    description: str = Field(description="Brief explanation of how this helps the candidate bridge their skill gap.")

class SkillGapReport(BaseModel):
    missing_skills: List[str] = Field(description="Key skills required or nice-to-have in the JD that are missing from the candidate's skills list.")
    learning_time_inferred: str = Field(description="Estimated time to close the gap (e.g. '~2 months'), inferred from candidate trajectory.")
    suggested_resources: List[SuggestedResource] = Field(description="List of 2 to 3 suggested learning resources.")

class CandidateExplanation(BaseModel):
    justification: str = Field(description="Exactly two-sentence match justification. Sentence 1: candidate skill/experience matching JD. Sentence 2: domain/seniority alignment (incorporating trajectory if label is High momentum or Steady growth). No metrics.")
    skill_gap_report: SkillGapReport = Field(description="Precisely identifies missing skills, estimated learning time, and suggestions to bridge the gap.")
    targeted_interview_questions: List[str] = Field(description="List of exactly 3 targeted technical or behavioral interview questions designed to probe the candidate on their specific skill gaps or transition areas compared to the job description.")


def generate_explanation(llm_client, jd_data: dict, candidate: dict) -> CandidateExplanation:
    """
    Constructs a customized prompt and calls the LLM with a structured schema to get
    both a 2-sentence match justification and a precise skill gap report with learning resources.
    """
    from langchain_core.prompts import ChatPromptTemplate
    
    # Extract candidate fields gracefully handling missing/null fields
    skills = candidate.get("skills")
    skills_str = ", ".join(skills) if isinstance(skills, list) and skills else "Not specified"
    prev_titles = candidate.get("previous_titles")
    prev_titles_str = ", ".join(prev_titles) if isinstance(prev_titles, list) and prev_titles else "None"
    
    # Extract trajectory fields for context
    trajectory_label = candidate.get("trajectory_label", "")
    trajectory_score = candidate.get("trajectory_score", 0.0)
    trajectory_signals = candidate.get("trajectory_signals", [])
    recent_skills = candidate.get("recent_skills_added", [])
    recent_skills_str = ", ".join(recent_skills) if recent_skills else "None"
    trajectory_signals_str = "; ".join(trajectory_signals) if trajectory_signals else "None"

    cand_details = (
        f"Name: {candidate.get('name')}\n"
        f"Current Title: {candidate.get('current_title', 'Not specified')}\n"
        f"Experience: {candidate.get('experience_years', 'Not specified')} years\n"
        f"Domain: {candidate.get('domain', 'Not specified')}\n"
        f"Skills: {skills_str}\n"
        f"Previous Titles: {prev_titles_str}\n"
        f"Summary/Bio: {candidate.get('bio', 'None')}\n"
        f"Career Trajectory: {trajectory_label} (score: {trajectory_score:.2f})\n"
        f"Trajectory Signals: {trajectory_signals_str}\n"
        f"Recently Added Skills: {recent_skills_str}\n"
    )
    
    # Standardize JD requirements for context
    jd_details = (
        f"Role Seniority: {jd_data.get('seniority', 'Not specified')}\n"
        f"Target Domain: {jd_data.get('domain', 'Not specified')}\n"
        f"Required Core Skills: {', '.join(jd_data.get('required_skills', []))}\n"
        f"Nice-To-Have Skills: {', '.join(jd_data.get('nice_to_have', []))}\n"
        f"Implicit Signals: {', '.join(jd_data.get('implicit_signals', []))}"
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a professional talent acquisition assistant. Your job is to output a structured CandidateExplanation JSON. "
            "Follow these constraints:\n"
            "- justification: exactly 2 sentences. Sentence 1 covers candidate technical skills/experience matching the JD. "
            "Sentence 2 covers domain/seniority alignment. If Career Trajectory is 'High momentum' or 'Steady growth', weave "
            "their upward momentum into the second sentence. Do not mention any scores or math.\n"
            "- skill_gap_report: missing_skills is a list of skills in the JD requirements that are missing from the candidate's skills list. "
            "learning_time_inferred is estimated time to close the gap based on candidate's trajectory (High momentum: ~1-2 months, "
            "Steady growth: ~2-3 months, Early stage: ~3-4 months, Plateau: ~5-6 months). "
            "suggested_resources is a list of 2-3 specific learning resources (e.g. online courses, tutorials, docs, github repos) to help them bridge the gap.\n"
            "- targeted_interview_questions: exactly 3 targeted technical or behavioral interview questions probing the candidate on their specific skill gaps or career transition areas."
        ),
        (
            "human",
            "Analyze this candidate and fill in the structured response:\n\n"
            "### Job Description Requirements ###\n{jd_details}\n\n"
            "### Candidate Profile ###\n{cand_details}"
        )
    ])
    
    # Generate structured explanation
    structured_llm = llm_client.with_structured_output(CandidateExplanation)
    chain = prompt | structured_llm
    
    response = chain.invoke({
        "jd_details": jd_details,
        "cand_details": cand_details
    })
    
    return response

def explain_top_candidates(jd_file: str, ranked_candidates_file: str, limit: int = 5) -> list:
    """
    Orchestrates Stage 3. Loads ranked candidates, runs LLM justifications for the top N,
    and updates the JSON file.
    """
    # 1. Load environment and check keys
    utils.load_environment()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.error("GROQ_API_KEY is not configured.")
        raise ValueError("Missing GROQ_API_KEY")
        
    # 2. Load inputs
    jd_data = utils.load_json(jd_file)
    candidates = utils.load_json(ranked_candidates_file)
    
    if not candidates:
        logger.error("No ranked candidates to explain.")
        raise ValueError("Ranked candidates list is empty.")

    # 3. Initialize LLM Client
    try:
        if api_key.startswith("xai-"):
            from langchain_openai import ChatOpenAI
            logger.info("Initializing xAI (Grok) Client for justifications...")
            llm = ChatOpenAI(
                model="grok-beta",
                temperature=0.3,
                api_key=api_key,
                base_url="https://api.x.ai/v1"
            )
        else:
            from langchain_groq import ChatGroq
            logger.info("Initializing Groq Client for justifications...")
            llm = ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0.3,
                api_key=api_key
            )
    except ImportError as e:
        logger.error("Failed to import LLM dependencies.")
        raise e

    # 4. Generate justifications for top N candidates
    total_to_process = min(limit, len(candidates))
    logger.info(f"Generating explanations for the top {total_to_process} candidates...")
    
    for i in range(total_to_process):
        candidate = candidates[i]
        logger.info(f"Processing explanation/gap report for {candidate['name']} (Rank {candidate['rank']})...")
        
        # Capture and preserve any existing recruiter_flag to ensure we don't overwrite it
        existing_flag = candidate.get("recruiter_flag")
        
        try:
            explanation: CandidateExplanation = generate_explanation(llm, jd_data, candidate)
            candidate["justification"] = explanation.justification
            candidate["skill_gap_report"] = explanation.skill_gap_report.model_dump()
            candidate["targeted_interview_questions"] = explanation.targeted_interview_questions
            logger.info(f"Generated justification: {explanation.justification}")
            logger.info(f"Generated skill gap: {candidate['skill_gap_report']}")
            logger.info(f"Generated interview questions: {candidate['targeted_interview_questions']}")
        except Exception as e:
            logger.error(f"Failed to generate explanation for {candidate['name']}: {str(e)}")
            candidate["justification"] = (
                f"{candidate['name']} shows matching technical skills in required competencies. "
                f"Their experience level fits the seniority and domain requirements."
            )
            candidate["skill_gap_report"] = {
                "missing_skills": [],
                "learning_time_inferred": "unknown",
                "suggested_resources": []
            }
            candidate["targeted_interview_questions"] = [
                f"How do you plan to quickly build expertise in the specific technology requirements of this role?",
                f"Describe a time when you successfully adapted to the {jd_data.get('domain', 'target')} domain.",
                f"How do you ensure code quality and system performance when working with microservices?"
            ]
            
        # Explicitly restore or retain the existing recruiter_flag alongside the new justification
        if existing_flag is not None:
            candidate["recruiter_flag"] = existing_flag
            
    # Save the updated list back to the ranked candidates file
    utils.save_json(candidates, ranked_candidates_file)
    logger.info("Stage 3 complete. Ranked candidates file updated with explanations.")
    
    return candidates

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    jd_json = os.path.join(base_dir, "data", "parsed_jd.json")
    ranked_json = os.path.join(base_dir, "output", "ranked_candidates.json")
    
    try:
        updated_candidates = explain_top_candidates(jd_json, ranked_json, limit=5)
        
        print("\n=== SUCCESS: TOP CANDIDATE JUSTIFICATIONS & SKILL GAPS ===")
        for cand in updated_candidates[:5]:
            print(f"Rank {cand['rank']}: {cand['name']}")
            print(f"  Score: {cand['match_score']}")
            print(f"  Justification: {cand.get('justification')}")
            gap = cand.get("skill_gap_report", {})
            if gap:
                print(f"  Missing Skills: {', '.join(gap.get('missing_skills', []))}")
                print(f"  Learning Pace: {gap.get('learning_time_inferred')}")
                print("  Suggested Resources:")
                for res in gap.get("suggested_resources", []):
                    print(f"    - {res.get('name')} ({res.get('url_or_platform')}): {res.get('description')}")
            print("-" * 50)
        print("=============================================\n")
        
    except Exception as e:
        print(f"\n[ERROR] Stage 3 failed to execute: {str(e)}", file=sys.stderr)
        sys.exit(1)
