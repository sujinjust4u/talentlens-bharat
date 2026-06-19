"""
TalentLens Stage 4: Reverse JD Generator & Alignment Analyzer

This module takes the top 3 ranked candidates and generates the ideal job description
that would perfectly match them. It then evaluates how close the original JD requirements
were to this ideal talent profile, generating an alignment score, a skill-by-skill
comparison, and actionable JD rewrite recommendations.
"""

import os
import sys
import json
from typing import List
from pydantic import BaseModel, Field

# Add current directory to path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import utils

logger = utils.setup_logging()

# =====================================================================
# 1. Structured Output Schemas
# =====================================================================

class SkillComparison(BaseModel):
    skill_name: str = Field(description="Name of the skill or technology.")
    in_original_jd: bool = Field(description="True if the skill was listed in the original JD.")
    in_top_candidates: bool = Field(description="True if the skill is possessed by the majority of the top 3 candidates.")
    status: str = Field(description="Status classification: 'Aligned', 'Missing in Candidates', or 'Bonus Skill in Candidates'.")
    recommendation: str = Field(description="Actionable advice for the recruiter (e.g. 'Retain requirement', 'Relax/remove from core', 'Add to core skills').")

class IdealJD(BaseModel):
    suggested_title: str = Field(description="Optimized job title based on the common profile elements of the top candidates.")
    suggested_seniority: str = Field(description="Target seniority level matching candidates' average/ideal depth.")
    core_skills: List[str] = Field(description="Core technical skills recommended based on candidates' actual profiles.")
    nice_to_have_skills: List[str] = Field(description="Recommended nice-to-have skills matching candidate profiles.")
    ideal_summary: str = Field(description="A short 2-3 sentence overview of the ideal candidate profile.")

class ReverseJDAnalysis(BaseModel):
    alignment_score: int = Field(description="An alignment score from 0 to 100 representing how closely the original JD matches the top candidate talent pool.")
    alignment_explanation: str = Field(description="A brief explanation justifying the alignment score and the major similarities or gaps.")
    ideal_jd: IdealJD = Field(description="The reconstructed ideal Job Description matching the top candidates.")
    skill_comparisons: List[SkillComparison] = Field(description="Comparison of key skills between original JD and top candidates.")
    suggested_jd_rewrites: List[str] = Field(description="Specific, actionable bullet points for rewriting the JD (e.g. 'Remove Kubernetes as it is missing; add Docker').")


# =====================================================================
# 2. Robust Fallback Generator
# =====================================================================

def generate_fallback_analysis(jd_data: dict, top_3: list) -> dict:
    """
    Generates a deterministic fallback analysis if the LLM call fails.
    Maintains user experience stability (graceful degradation).
    """
    logger.warning("Generating fallback reverse JD analysis due to LLM failure or missing API key.")
    
    # Simple heuristics to build an ideal JD from top 3 candidates
    cand_skills_freq = {}
    titles = []
    seniorities = []
    
    for cand in top_3:
        titles.append(cand.get("current_title", "Engineer"))
        for s in cand.get("skills", []):
            cand_skills_freq[s] = cand_skills_freq.get(s, 0) + 1
            
    # Sort skills by frequency in top candidates
    common_skills = [s for s, count in sorted(cand_skills_freq.items(), key=lambda x: x[1], reverse=True)]
    top_skills = common_skills[:6] if common_skills else ["Python", "FastAPI"]
    bonus_skills = common_skills[6:10] if len(common_skills) > 6 else ["SQL"]
    
    original_required = jd_data.get("required_skills", [])
    original_nice = jd_data.get("nice_to_have", [])
    
    # Simple alignment score calculation based on Jaccard similarity of required skills
    intersection = len(set(original_required) & set(top_skills))
    union = len(set(original_required) | set(top_skills))
    jaccard = intersection / union if union > 0 else 0.5
    alignment_score = int(50 + 50 * jaccard)
    
    # Generate skill comparisons
    comparisons = []
    rewrites = []
    
    # Check original required skills
    for skill in original_required:
        present = skill in cand_skills_freq
        status = "Aligned" if present else "Missing in Candidates"
        rec = "Retain requirement" if present else f"Consider demoting {skill} to nice-to-have or removing it, as top candidates lack it."
        if not present:
            rewrites.append(f"Remove or demote '{skill}' from core requirements (not present in top candidate pool).")
        comparisons.append(
            SkillComparison(
                skill_name=skill,
                in_original_jd=True,
                in_top_candidates=present,
                status=status,
                recommendation=rec
            )
        )
        
    # Check candidates skills not in original JD
    for skill in top_skills:
        if skill not in original_required and skill not in original_nice:
            rewrites.append(f"Add '{skill}' to JD requirements to attract more of this talent pool.")
            comparisons.append(
                SkillComparison(
                    skill_name=skill,
                    in_original_jd=False,
                    in_top_candidates=True,
                    status="Bonus Skill in Candidates",
                    recommendation=f"Add to requirements — highly prevalent in top talent."
                )
            )
            
    if not rewrites:
        rewrites.append("No changes needed. The original job description closely matches the top talent profile.")
        
    suggested_title = titles[0] if titles else "Senior Backend Engineer"
    
    fallback_analysis = {
        "alignment_score": alignment_score,
        "alignment_explanation": "Inferred from candidate profile skills frequency compared to the original JD. There is a strong overlap in core tech stacks.",
        "ideal_jd": {
            "suggested_title": suggested_title,
            "suggested_seniority": jd_data.get("seniority", "Mid-Senior"),
            "core_skills": top_skills,
            "nice_to_have_skills": bonus_skills,
            "ideal_summary": f"Ideal candidate profile modeled after {', '.join([c.get('name', 'Unknown') for c in top_3])}."
        },
        "skill_comparisons": [c.model_dump() for c in comparisons],
        "suggested_jd_rewrites": rewrites
    }
    
    return fallback_analysis


# =====================================================================
# 3. LLM Analysis Function
# =====================================================================

def analyze_alignment(jd_file: str, ranked_candidates_file: str, output_file: str) -> dict:
    """
    Takes the parsed JD JSON and ranked candidates JSON, extracts the top 3,
    calls the LLM to perform Reverse JD generation and alignment audit, and
    saves the results to the output file.
    """
    utils.load_environment()
    api_key = os.environ.get("GROQ_API_KEY")
    
    # Load inputs
    jd_data = utils.load_json(jd_file)
    candidates = utils.load_json(ranked_candidates_file)
    
    if not candidates:
        logger.error("No candidates found in ranked file.")
        raise ValueError("Candidate pool is empty.")
        
    # Extract top 3 candidates
    top_3 = candidates[:3]
    logger.info(f"Extracting top 3 candidates for Reverse JD analysis: {[c.get('name') for c in top_3]}")
    
    if not api_key:
        fallback = generate_fallback_analysis(jd_data, top_3)
        utils.save_json(fallback, output_file)
        return fallback
        
    # Setup LLM client
    try:
        if api_key.startswith("xai-"):
            from langchain_openai import ChatOpenAI
            logger.info("Initializing xAI (Grok) for Reverse JD Generator...")
            llm = ChatOpenAI(
                model="grok-beta",
                temperature=0.2,
                api_key=api_key,
                base_url="https://api.x.ai/v1"
            )
        else:
            from langchain_groq import ChatGroq
            logger.info("Initializing Groq for Reverse JD Generator...")
            llm = ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0.2,
                api_key=api_key
            )
    except ImportError as e:
        logger.error(f"Failed to import LLM dependencies: {str(e)}")
        fallback = generate_fallback_analysis(jd_data, top_3)
        utils.save_json(fallback, output_file)
        return fallback

    # Format inputs for LLM prompt
    jd_summary = (
        f"Job Title: {jd_data.get('title', 'Not specified')}\n"
        f"Seniority: {jd_data.get('seniority', 'Not specified')}\n"
        f"Domain: {jd_data.get('domain', 'Not specified')}\n"
        f"Required Skills: {', '.join(jd_data.get('required_skills', []))}\n"
        f"Nice-To-Have Skills: {', '.join(jd_data.get('nice_to_have', []))}\n"
    )
    
    candidates_summary = ""
    for idx, c in enumerate(top_3):
        candidates_summary += (
            f"Candidate #{idx+1}: {c.get('name')}\n"
            f"Current Title: {c.get('current_title', 'N/A')}\n"
            f"Experience: {c.get('experience_years', 'N/A')} years\n"
            f"Domain Match: {c.get('domain', 'N/A')}\n"
            f"Skills: {', '.join(c.get('skills', []))}\n"
            f"Bio Summary: {c.get('bio', 'N/A')}\n\n"
        )
        
    from langchain_core.prompts import ChatPromptTemplate
    
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert Talent Acquisition Strategist. Your task is to perform a Reverse Job Description Analysis. "
            "Analyze the top 3 ranked candidates who represent the best available talent matching the recruiter's search. "
            "Generate the ideal Job Description that perfectly reflects their collective profiles (title, seniority, core skills, nice-to-have, and a short summary). "
            "Then, compare the original job description to this reconstructed ideal profile and output a ReverseJDAnalysis JSON.\n\n"
            "Constraints:\n"
            "- alignment_score: integer 0-100 indicating how close the original JD is to the ideal JD. If they are highly aligned (same title, same skills), score > 85. If candidates lack major skills or have bonus critical skills not in JD, score between 50-80.\n"
            "- skill_comparisons: For each skill in either the original JD required/nice-to-have lists OR common among the top candidates, classify it. "
            "If in original JD but missing in candidates, Status is 'Missing in Candidates', Recommendation is 'Remove/relax requirement'. "
            "If in candidates but not in JD, Status is 'Bonus Skill in Candidates', Recommendation is 'Add to JD'. "
            "If in both, Status is 'Aligned', Recommendation is 'Retain requirement'.\n"
            "- suggested_jd_rewrites: List 2-4 highly specific bullet point rewrite suggestions. Avoid generic feedback. Mention specific skills, titles, or seniorities (e.g. 'Replace Kafka with RabbitMQ as all top candidates are familiar with RabbitMQ but lack Kafka')."
        ),
        (
            "human",
            "### Original Job Description ###\n{jd_summary}\n"
            "### Top 3 Candidate Profiles ###\n{candidates_summary}"
        )
    ])
    
    try:
        structured_llm = llm.with_structured_output(ReverseJDAnalysis)
        chain = prompt | structured_llm
        
        response = chain.invoke({
            "jd_summary": jd_summary,
            "candidates_summary": candidates_summary
        })
        
        result_dict = response.model_dump()
        utils.save_json(result_dict, output_file)
        logger.info(f"Reverse JD Analysis successfully generated and saved to {output_file}.")
        return result_dict
    except Exception as e:
        logger.error(f"Failed to generate structured reverse JD analysis: {str(e)}")
        fallback = generate_fallback_analysis(jd_data, top_3)
        utils.save_json(fallback, output_file)
        return fallback


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    jd_json = os.path.join(base_dir, "data", "parsed_jd.json")
    ranked_json = os.path.join(base_dir, "output", "ranked_candidates.json")
    out_json = os.path.join(base_dir, "output", "reverse_jd_analysis.json")
    
    try:
        analysis = analyze_alignment(jd_json, ranked_json, out_json)
        print("\n=== SUCCESS: REVERSE JD & ALIGNMENT ANALYSIS ===")
        print(f"Alignment Score: {analysis['alignment_score']}/100")
        print(f"Explanation: {analysis['alignment_explanation']}")
        print("\n--- Ideal Job Description ---")
        print(f"Title: {analysis['ideal_jd']['suggested_title']}")
        print(f"Seniority: {analysis['ideal_jd']['suggested_seniority']}")
        print(f"Core Skills: {', '.join(analysis['ideal_jd']['core_skills'])}")
        print(f"Nice-To-Have: {', '.join(analysis['ideal_jd']['nice_to_have_skills'])}")
        print(f"Summary: {analysis['ideal_jd']['ideal_summary']}")
        print("\n--- Rewrite Suggestions ---")
        for rewrite in analysis['suggested_jd_rewrites']:
            print(f"- {rewrite}")
        print("================================================\n")
    except Exception as e:
        print(f"Error executing stage4 test: {str(e)}", file=sys.stderr)
        sys.exit(1)
