#!/usr/bin/env python3
"""
TalentLens Bharat Pipeline Evaluator

In production, this ground truth would come from recruiter accept/reject feedback (telemetry).
This script validates the pipeline's ranking performance against human-labeled true positives
before deploying the pipeline configuration to a new job category.

Usage:
    python3 evaluate.py
"""

import json
import os
import sys

# Define ground truth of true positives (strong matches for the Senior Backend Engineer (AI & Data Platforms) role).
# These candidates have 5+ years of experience, a Senior title, the target Fintech domain, and the core required technical skills.
GROUND_TRUTH_STRONG_MATCHES = {
    "cand_01",  # Aarav Sharma (Senior Backend Engineer, Fintech, 6 yrs, Python, FastAPI, K8s, Redis, LangChain)
    "cand_08",  # Rohit Verma (Senior Backend Engineer, Fintech, 6 yrs, Python, FastAPI, K8s/Docker, Redis)
    "cand_10",  # Sandeep Gupta (Senior Backend Engineer, Fintech, 6 yrs, Python, PostgreSQL, Docker, K8s, FastAPI)
    "cand_19",  # Rajesh Iyer (Senior Backend Engineer, Fintech, 7 yrs, Python, Redis, LangChain, Hugging Face, K8s, FastAPI)
    "cand_20",  # Preeti Deshmukh (Senior Backend Engineer, Fintech, 8 yrs, Python, FastAPI, Redis, LangChain, K8s)
}

def load_ranked_results(filepath: str) -> list:
    """Loads the candidate results from ranked_candidates.json."""
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found. Please run the pipeline first using 'python3 main.py'.", file=sys.stderr)
        sys.exit(1)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading ranked candidates: {str(e)}", file=sys.stderr)
        sys.exit(1)

def compute_precision_at_k(ranked_candidates: list, k: int, ground_truth: set) -> float:
    """Computes the Precision@K metric."""
    if not ranked_candidates or k <= 0:
        return 0.0
    
    # Take the top K candidates
    top_k = ranked_candidates[:k]
    
    # Count how many of these are in the ground truth
    true_positives_count = sum(1 for cand in top_k if cand.get("id") in ground_truth)
    
    return true_positives_count / min(len(ranked_candidates), k)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ranked_file = os.path.join(base_dir, "output", "ranked_candidates.json")
    
    print("================================================================================")
    print("                     TALENTLENS PIPELINE RANKING EVALUATION")
    print("================================================================================")
    print(f"Loading ranked results from: {ranked_file}")
    
    ranked_candidates = load_ranked_results(ranked_file)
    print(f"Loaded {len(ranked_candidates)} candidates.")
    print("-" * 80)
    
    # Compute Precision at different thresholds
    k_values = [3, 5, 10]
    results = {}
    
    for k in k_values:
        precision = compute_precision_at_k(ranked_candidates, k, GROUND_TRUTH_STRONG_MATCHES)
        results[k] = precision
        
    # Print the evaluation table
    print(f"{'Metric':<15} | {'Target K':<10} | {'True Positives in Top K':<25} | {'Precision@K (%)':<15}")
    print("-" * 80)
    
    for k in k_values:
        top_k = ranked_candidates[:k]
        tp_found = sum(1 for cand in top_k if cand.get("id") in GROUND_TRUTH_STRONG_MATCHES)
        precision_pct = results[k] * 100.0
        print(f"Precision@{k:<6} | {k:<10} | {tp_found:<25} | {precision_pct:<15.1f}%")
        
    print("-" * 80)
    print("Top 5 Candidates in Ranking:")
    for idx, cand in enumerate(ranked_candidates[:5]):
        cand_id = cand.get("id")
        name = cand.get("name")
        title = cand.get("current_title")
        score = cand.get("final_score")
        is_tp = "✓ True Positive" if cand_id in GROUND_TRUTH_STRONG_MATCHES else "✗ False Positive"
        print(f"  Rank {idx+1}: {name} ({title}) | Final Score: {score:.4f} | Status: {is_tp}")
        
    print("================================================================================")

if __name__ == "__main__":
    main()
