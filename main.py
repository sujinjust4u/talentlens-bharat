import os
import sys
import time
import argparse

# Add current directory to path to import stages
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import utils
import stage1_parser
import stage2_ranker
from stage2b_india_signals import apply_india_signals
from stage2c_trajectory import apply_trajectory
from stage2d_clustering import cluster_candidates
from stage2e_bias_audit import run_fairness_audit
import stage3_explainer
import stage4_reverse_jd

logger = utils.setup_logging()

def run_pipeline(jd_file: str, candidates_file: str, output_file: str, explanation_limit: int = 5) -> None:
    """
    Orchestrates the entire candidate discovery and ranking pipeline.
    Measures duration and logs progress for each module.
    """
    start_time = time.time()
    logger.info("Initializing TalentLens pipeline execution...")

    # Ensure project workspace structure exists
    base_dir = os.path.dirname(os.path.abspath(__file__))
    utils.ensure_directories_exist([
        os.path.join(base_dir, "data"),
        os.path.join(base_dir, "output")
    ])

    # Define intermediate file path for parsed JD
    parsed_jd_file = os.path.join(base_dir, "data", "parsed_jd.json")

    # -------------------------------------------------------------
    # Stage 1: LLM Job Description Parser
    # -------------------------------------------------------------
    logger.info("--- STARTING STAGE 1: Job Description Parsing ---")
    s1_start = time.time()
    try:
        stage1_parser.parse_job_description(jd_file, parsed_jd_file)
        logger.info(f"Stage 1 completed in {time.time() - s1_start:.2f} seconds.")
    except Exception as e:
        logger.error(f"Pipeline aborted. Stage 1 Parser failed: {str(e)}")
        sys.exit(1)

    # -------------------------------------------------------------
    # Stage 2: Semantic Similarity Ranker
    # -------------------------------------------------------------
    logger.info("--- STARTING STAGE 2: Semantic Ranking ---")
    s2_start = time.time()
    try:
        ranked_candidates = stage2_ranker.rank_candidates(parsed_jd_file, candidates_file, output_file)
        logger.info(f"Stage 2 completed in {time.time() - s2_start:.2f} seconds.")
    except Exception as e:
        logger.error(f"Pipeline aborted. Stage 2 Ranker failed: {str(e)}")
        sys.exit(1)

    # -------------------------------------------------------------
    # Stage 2b: India-Specific Re-ranking Signals
    # -------------------------------------------------------------
    ranked_candidates = apply_india_signals(ranked_candidates); utils.save_json(ranked_candidates, output_file)

    # -------------------------------------------------------------
    # Stage 2c: Career Trajectory Scoring
    # -------------------------------------------------------------
    logger.info("--- STARTING STAGE 2c: Career Trajectory Prediction ---")
    s2c_start = time.time()
    try:
        ranked_candidates = apply_trajectory(ranked_candidates)
        utils.save_json(ranked_candidates, output_file)
        logger.info(f"Stage 2c completed in {time.time() - s2c_start:.2f} seconds.")
    except Exception as e:
        logger.error(f"Stage 2c warning: {str(e)}")

    # -------------------------------------------------------------
    # Stage 2d: Candidate Persona Clustering
    # -------------------------------------------------------------
    logger.info("--- STARTING STAGE 2d: Candidate Persona Clustering ---")
    s2d_start = time.time()
    cluster_rec_text = "Clustering not run."
    try:
        parsed_jd_data = utils.load_json(parsed_jd_file)
        clustering_res = cluster_candidates(parsed_jd_data, ranked_candidates)
        cluster_assignments = clustering_res.get("candidate_clusters", {})
        cluster_rec_text = clustering_res.get("cluster_recommendation", "Clustering complete.")
        
        # Enrich candidate profiles with persona clusters
        for cand in ranked_candidates:
            cand["persona_cluster"] = cluster_assignments.get(cand["id"], "The Generalist Builder")
            
        utils.save_json(ranked_candidates, output_file)
        logger.info(f"Stage 2d completed in {time.time() - s2d_start:.2f} seconds.")
    except Exception as e:
        logger.error(f"Stage 2d warning: {str(e)}")

    # -------------------------------------------------------------
    # Stage 2e: Bias Detection & Fairness Audit
    # -------------------------------------------------------------
    logger.info("--- STARTING STAGE 2e: Bias Detection & Fairness Audit ---")
    s2e_start = time.time()
    fairness_report = None
    try:
        fairness_report = run_fairness_audit(ranked_candidates)
        logger.info(f"Stage 2e completed in {time.time() - s2e_start:.2f} seconds.")
    except Exception as e:
        logger.error(f"Stage 2e warning: {str(e)}")

    # -------------------------------------------------------------
    # Stage 3: LLM Explainer (Top N Candidates)
    # -------------------------------------------------------------
    logger.info(f"--- STARTING STAGE 3: Generating Justifications (Top {explanation_limit}) ---")
    s3_start = time.time()
    try:
        final_candidates = stage3_explainer.explain_top_candidates(
            parsed_jd_file, 
            output_file, 
            limit=explanation_limit
        )
        logger.info(f"Stage 3 completed in {time.time() - s3_start:.2f} seconds.")
    except Exception as e:
        logger.error(f"Pipeline finished with Stage 3 warning: {str(e)}")
        # We don't exit here as Stage 2 ranking is already saved, and justifications are value-add.
        # This is a classic graceful degradation strategy.
        final_candidates = utils.load_json(output_file)

    # -------------------------------------------------------------
    # Advanced Intelligence Enrichment: Confidence Reasoning & Dark Horse
    # -------------------------------------------------------------
    logger.info("Enriching candidates with structured confidence reasoning...")
    parsed_jd_data = utils.load_json(parsed_jd_file)
    for cand in final_candidates:
        cand["confidence_reasoning"] = utils.compute_confidence_reasoning(cand, parsed_jd_data)
        
    utils.save_json(final_candidates, output_file)
    
    # Identify Dark Horse candidate
    dark_horse_candidate = utils.select_dark_horse(final_candidates, limit=explanation_limit)

    # -------------------------------------------------------------
    # Stage 4: Reverse JD Generator & Alignment Analyzer
    # -------------------------------------------------------------
    logger.info("--- STARTING STAGE 4: Reverse JD Generator & Alignment Analyzer ---")
    s4_start = time.time()
    reverse_jd_file = output_file.replace(".json", "_reverse_jd.json")
    reverse_jd_report = None
    try:
        reverse_jd_report = stage4_reverse_jd.analyze_alignment(
            parsed_jd_file, 
            output_file, 
            reverse_jd_file
        )
        logger.info(f"Stage 4 completed in {time.time() - s4_start:.2f} seconds.")
    except Exception as e:
        logger.error(f"Stage 4 warning: {str(e)}")

    # -------------------------------------------------------------
    # Execution Summary
    # -------------------------------------------------------------
    total_duration = time.time() - start_time
    logger.info(f"TalentLens Pipeline executed successfully in {total_duration:.2f} seconds.")

    print("\n" + "=" * 80)
    print("                      TALENTLENS PIPELINE EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Input Job Description : {jd_file}")
    print(f"Input Candidate Pool  : {candidates_file}")
    print(f"Output Ranked Results : {output_file}")
    print(f"Total Execution Time  : {total_duration:.2f} seconds")
    print("-" * 80)
    print(f"RECRUITER RECOMMENDATION:\n{cluster_rec_text}")
    print("-" * 80)
    
    # Print Fairness Report Summary
    if fairness_report:
        print("FAIRNESS REPORT:")
        print(f"  Score: {fairness_report.get('fairness_score', 'N/A')}/100 (Grade: {fairness_report.get('fairness_grade', 'N/A')})")
        for dim in fairness_report.get("dimensions", []):
            verdict = dim.get("verdict", {})
            print(f"  {verdict.get('summary', '')}")
            underranked = dim.get("potentially_underranked", [])
            if underranked:
                for flag in underranked:
                    print(f"    ↳ {flag.get('reason', '')}")
        print("-" * 80)
        
    # Print Reverse JD Report Summary
    if reverse_jd_report:
        print("REVERSE JD ALIGNMENT REPORT:")
        print(f"  Alignment Score: {reverse_jd_report.get('alignment_score', 'N/A')}/100")
        print(f"  Explanation: {reverse_jd_report.get('alignment_explanation', 'N/A')}")
        ideal = reverse_jd_report.get("ideal_jd", {})
        if ideal:
            print("  --- Reconstructed Ideal JD ---")
            print(f"    Title: {ideal.get('suggested_title', 'N/A')}")
            print(f"    Seniority: {ideal.get('suggested_seniority', 'N/A')}")
            print(f"    Core Skills: {', '.join(ideal.get('core_skills', []))}")
            print(f"    Summary: {ideal.get('ideal_summary', 'N/A')}")
        rewrites = reverse_jd_report.get("suggested_jd_rewrites", [])
        if rewrites:
            print("  --- Recruiter JD Rewrite Suggestions ---")
            for rw in rewrites:
                print(f"    ↳ {rw}")
        print("-" * 80)
        
    # Print Dark Horse Spotlight
    if 'dark_horse_candidate' in locals() and dark_horse_candidate:
        print("🏇 DARK HORSE SPOTLIGHT:")
        print(f"  Candidate: {dark_horse_candidate.get('name')} (Rank {dark_horse_candidate.get('rank')}, Score: {dark_horse_candidate.get('final_score'):.4f}) — {dark_horse_candidate.get('current_title')}")
        print(f"  ↳ Potential Reason: {dark_horse_candidate.get('reason')}")
        print("-" * 80)
        
    print("TOP CANDIDATES AND JUSTIFICATIONS:")
    print("-" * 80)
    
    for cand in final_candidates:
        rank = cand.get("rank", 99)
        name = cand.get("name", "Unknown")
        score = cand.get("match_score", 0.0)
        title = cand.get("current_title", "N/A")
        justification = cand.get("justification")
        recruiter_flag = cand.get("recruiter_flag")
        india_signals = cand.get("india_signals_detected", [])
        
        # Format printing dynamically
        cluster = cand.get("persona_cluster", "The Generalist Builder")
        confidence_text = cand.get("confidence_reasoning", f"Score: {score}")
        print(f"Rank {rank}: {name} ({title}) | {confidence_text} | Persona: {cluster}")
        if india_signals:
            print(f"  India Signals: {', '.join(india_signals)}")
        trajectory_label = cand.get("trajectory_label")
        trajectory_score = cand.get("trajectory_score", 0.0)
        trajectory_signals = cand.get("trajectory_signals", [])
        if trajectory_label:
            print(f"  Trajectory: {trajectory_label} ({trajectory_score:.2f})")
        if trajectory_signals:
            for sig in trajectory_signals:
                print(f"    ↳ {sig}")
        if recruiter_flag:
            print(f"  [FLAG] {recruiter_flag}")
        if justification:
            print(f"  Justification: {justification}")
        else:
            print("  Justification: Not generated.")
            
        # Skill Gap Report & Learning Bridge
        gap_report = cand.get("skill_gap_report")
        if gap_report:
            missing = gap_report.get("missing_skills", [])
            pace = gap_report.get("learning_time_inferred", "")
            resources = gap_report.get("suggested_resources", [])
            if missing:
                print(f"  Missing Skills: {', '.join(missing)}")
            if pace:
                print(f"  Learning Pace (Inferred): {pace}")
            if resources:
                print("  Suggested Resources:")
                for r in resources:
                    print(f"    ↳ {r.get('name')} ({r.get('url_or_platform')}): {r.get('description')}")
                    
        # Targeted Interview Questions
        questions = cand.get("targeted_interview_questions", [])
        if questions:
            print("  Targeted Interview Questions:")
            for q in questions:
                print(f"    ↳ {q}")
        print("-" * 80)
    print("=" * 80 + "\n")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set default file paths relative to this script directory
    default_jd = os.path.join(base_dir, "data", "job_description.txt")
    default_cand = os.path.join(base_dir, "data", "candidates.json")
    default_output = os.path.join(base_dir, "output", "ranked_candidates.json")

    parser = argparse.ArgumentParser(
        description="TalentLens Intelligent Candidate Discovery Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--jd", 
        type=str, 
        default=default_jd, 
        help="Path to raw job description text file."
    )
    parser.add_argument(
        "--candidates", 
        type=str, 
        default=default_cand, 
        help="Path to candidate pool JSON file."
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=default_output, 
        help="Path to save output ranked JSON file."
    )
    parser.add_argument(
        "--explain-limit", 
        type=int, 
        default=5, 
        help="Number of top candidates to generate LLM justifications for."
    )

    args = parser.parse_args()
    
    run_pipeline(
        jd_file=args.jd, 
        candidates_file=args.candidates, 
        output_file=args.output,
        explanation_limit=args.explain_limit
    )
