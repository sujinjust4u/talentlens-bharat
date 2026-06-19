import os
import sys
from typing import List, Optional
from pydantic import BaseModel, Field

# Add current directory to path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import utils

# Initialize Logger
logger = utils.setup_logging()

# =====================================================================
# 1. Structured Data Schema Design
# =====================================================================
class ParsedJobDescription(BaseModel):
    """
    Structured schema representing the essential aspects of a Job Description.
    This structure is fed into the downstream Semantic Ranker (Stage 2).
    """
    required_skills: List[str] = Field(
        description="List of mandatory technical and soft skills, tools, frameworks, and programming languages required."
    )
    nice_to_have: List[str] = Field(
        description="Preferred but non-mandatory skills, technologies, databases, cloud providers, or certifications."
    )
    seniority: str = Field(
        description="Target seniority level of the position. E.g., Junior, Mid, Senior, Lead, Principal, Intern, or 'Not Specified'."
    )
    domain: str = Field(
        description="The industry domain or technical focus. E.g., Fintech, Healthtech, E-commerce, SaaS, AI/ML, Cybersecurity, General."
    )
    implicit_signals: List[str] = Field(
        description="Implicit expectations inferred from the description text. E.g., 'remote-friendly', 'startup experience', 'high-growth environment', 'database scaling', 'independent worker'."
    )


# =====================================================================
# 2. Stage 1 Parser Execution Logic
# =====================================================================
def parse_job_description(jd_filepath: str, output_filepath: str) -> dict:
    """
    Reads a raw text job description, invokes ChatGroq with a structured schema,
    and saves the parsed profile to a JSON file.
    """
    # Load environment variables (contains GROQ_API_KEY)
    utils.load_environment()
    
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.error("GROQ_API_KEY is not configured. Please define it in your environment or a .env file.")
        raise ValueError("Missing GROQ_API_KEY")
    
    # Read the raw JD file
    if not os.path.exists(jd_filepath):
        logger.error(f"Job Description file not found at: {jd_filepath}")
        raise FileNotFoundError(f"Missing file: {jd_filepath}")
        
    with open(jd_filepath, "r", encoding="utf-8") as f:
        raw_jd_text = f.read().strip()
        
    if not raw_jd_text:
        logger.error("Job description text file is empty.")
        raise ValueError("Job description file is empty.")
        
    logger.info(f"Loaded job description from: {jd_filepath} ({len(raw_jd_text)} chars)")
    
    # Initialize LangChain imports locally to isolate dependency issues
    try:
        from langchain_core.prompts import ChatPromptTemplate
        if api_key.startswith("xai-"):
            from langchain_openai import ChatOpenAI
            logger.info("Detected xAI API key prefix. Initializing xAI (Grok) client.")
            llm = ChatOpenAI(
                model="grok-beta",
                temperature=0.0,
                api_key=api_key,
                base_url="https://api.x.ai/v1",
                max_retries=3
            )
        else:
            from langchain_groq import ChatGroq
            logger.info("Detected Groq API key prefix. Initializing Groq client.")
            llm = ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0.0,
                api_key=api_key,
                max_retries=3
            )
    except ImportError as e:
        logger.error("Failed to import LangChain packages. Make sure to run 'pip install -r requirements.txt'")
        raise e
    
    # Create System and Human Chat Prompts
    # Explicit guidelines are given to ensure the output aligns perfectly with candidate profile matching.
    prompt_template = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert AI recruiting assistant specializing in candidate screening. "
            "Your task is to analyze the provided raw Job Description (JD) text and extract "
            "structured parameters for downstream applicant matching.\n\n"
            "Guidelines:\n"
            "1. Extract skills clean and standardized (e.g., 'Python', 'PostgreSQL', not 'Strong proficiency in Python').\n"
            "2. Infer seniority (e.g., 'Senior' if 5+ years experience or lead responsibilities are specified).\n"
            "3. Infer implicit signals such as working environment, scaling needs, or startup culture.\n"
            "4. Ensure output complies strictly with the schema structure."
        ),
        ("human", "Analyze this Job Description:\n\n{job_description}")
    ])
    
    # Bind the structured output schema to the LLM
    try:
        structured_llm = llm.with_structured_output(ParsedJobDescription)
        parser_chain = prompt_template | structured_llm
        
        logger.info("Parsing Job Description with LLM...")
        parsed_result: ParsedJobDescription = parser_chain.invoke({"job_description": raw_jd_text})
        
        # Convert Pydantic object to dictionary
        parsed_data = parsed_result.model_dump()
        
    except Exception as e:
        logger.error(f"Error during structured LLM parsing: {str(e)}")
        logger.info("Attempting fallback text parsing...")
        # A robust fallback: if structured tool-calling fails, we could request JSON directly,
        # or propagate the exception gracefully. Here we propagate to maintain type integrity.
        raise RuntimeError("Failed to structured-parse the JD using LLM. Please check your Groq API status/limits.") from e
        
    # Write the output using our utility function
    utils.save_json(parsed_data, output_filepath)
    logger.info("Stage 1 complete: Job Description successfully parsed.")
    
    return parsed_data

if __name__ == "__main__":
    # Define absolute/relative paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    jd_file = os.path.join(base_dir, "data", "job_description.txt")
    output_file = os.path.join(base_dir, "data", "parsed_jd.json")
    
    try:
        parsed_profile = parse_job_description(jd_file, output_file)
        
        # Print results formatted for easy verification
        print("\n=== SUCCESS: PARSED JOB DESCRIPTION PROFILE ===")
        import json
        print(json.dumps(parsed_profile, indent=4))
        print("================================================\n")
        
    except Exception as e:
        print(f"\n[ERROR] Stage 1 failed to execute: {str(e)}", file=sys.stderr)
        print("Please ensure your .env file is configured with GROQ_API_KEY.\n", file=sys.stderr)
        sys.exit(1)
