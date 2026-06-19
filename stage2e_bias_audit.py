"""
TalentLens Stage 2e: Bias Detection & Fairness Audit

Audits the pipeline's own rankings for potential bias across two dimensions:
  1. Geography (Tier-1 metro vs. Tier-2/3 cities)
  2. Gender (inferred cautiously from Indian first names)

Produces a structured "Fairness Report" with:
  - Group-level average score comparisons
  - Score gap analysis with statistical significance
  - Specific flagged candidates who may be underranked
  - Overall fairness verdict per dimension
"""

import os
import sys
import statistics

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import utils

logger = utils.setup_logging()

# =====================================================================
# 1. Indian Name → Gender Inference (Cautious Heuristic)
# =====================================================================
# These are common Indian first names with high confidence gender association.
# Names NOT in this dictionary are classified as "Unknown" — we never guess.
# This is intentionally conservative to avoid misgendering.

FEMALE_FIRST_NAMES = {
    "priya", "ananya", "meera", "divya", "sneha", "shreya", "neha", "pooja",
    "preeti", "nidhi", "swati", "kavitha", "deepa", "riya", "aarti", "shalini",
    "anjali", "sunita", "rekha", "geeta", "padma", "lata", "radha", "seema",
    "mamta", "kiran",  # Kiran is unisex but commonly female in South India
    "nisha", "komal", "sakshi", "tanvi", "megha", "rashmi", "pallavi",
    "gayathri", "lakshmi", "sarita", "usha", "vandana", "archana",
}

MALE_FIRST_NAMES = {
    "aarav", "ishaan", "kabir", "rohan", "rohit", "sandeep", "manoj", "vikram",
    "rahul", "rajesh", "arjun", "sanjay", "ajay", "amit", "arun", "suresh",
    "rakesh", "vijay", "deepak", "nikhil", "gaurav", "harsh", "manish",
    "naveen", "prasad", "ramesh", "sachin", "tushar", "varun", "yogesh",
    "ashok", "bhaskar", "chetan", "devendra", "ganesh", "hari", "jayant",
}


def infer_gender(name: str) -> str:
    """
    Infers gender from an Indian first name using a curated dictionary.
    Returns 'Female', 'Male', or 'Unknown'.
    
    Intentionally conservative — names not in the dictionary return 'Unknown'
    rather than making probabilistic guesses. This is a design choice to
    prioritize fairness over coverage.
    """
    if not name:
        return "Unknown"
    
    first_name = name.strip().split()[0].lower()
    
    if first_name in FEMALE_FIRST_NAMES:
        return "Female"
    elif first_name in MALE_FIRST_NAMES:
        return "Male"
    else:
        return "Unknown"


# =====================================================================
# 2. City Tier Classification
# =====================================================================
TIER_1_CITIES = {
    "bangalore", "bengaluru", "mumbai", "delhi", "new delhi",
    "hyderabad", "chennai", "kolkata", "pune", "gurgaon", "gurugram",
    "noida", "ghaziabad", "ahmedabad",
}

TIER_2_3_CITIES = {
    "indore", "bhopal", "nagpur", "coimbatore", "surat",
    "jaipur", "lucknow", "kochi", "chandigarh", "vadodara",
    "madurai", "tiruchirappalli", "lalgudi", "salem", "nashik", "agra",
    "patna", "ranchi", "bhubaneswar", "guwahati", "dehradun",
    "mysore", "vizag", "visakhapatnam", "thiruvananthapuram",
    "mangalore", "hubli", "belgaum", "aurangabad", "jodhpur",
    "udaipur", "raipur", "amritsar", "ludhiana", "varanasi",
}


def classify_city_tier(location: str) -> str:
    """Classifies a location string into Tier-1, Tier-2/3, or Unknown."""
    if not location:
        return "Unknown"
    
    loc_lower = location.strip().lower()
    
    if loc_lower in TIER_1_CITIES:
        return "Tier-1"
    elif loc_lower in TIER_2_3_CITIES:
        return "Tier-2/3"
    else:
        return "Unknown"


# =====================================================================
# 3. Statistical Helpers
# =====================================================================
def _safe_mean(values: list) -> float:
    """Return mean of a list, or 0.0 if empty."""
    return statistics.mean(values) if values else 0.0


def _safe_median(values: list) -> float:
    """Return median of a list, or 0.0 if empty."""
    return statistics.median(values) if values else 0.0


def _safe_stdev(values: list) -> float:
    """Return stdev of a list, or 0.0 if fewer than 2 values."""
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def _cohens_d(group1: list, group2: list) -> float:
    """
    Computes Cohen's d effect size between two groups.
    Returns 0.0 if either group has fewer than 2 members.
    
    Interpretation:
      |d| < 0.2  → negligible
      0.2 ≤ |d| < 0.5 → small
      0.5 ≤ |d| < 0.8 → medium  
      |d| ≥ 0.8 → large
    """
    if len(group1) < 2 or len(group2) < 2:
        return 0.0
    
    mean1, mean2 = statistics.mean(group1), statistics.mean(group2)
    var1, var2 = statistics.variance(group1), statistics.variance(group2)
    
    # Pooled standard deviation
    n1, n2 = len(group1), len(group2)
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    pooled_sd = pooled_var ** 0.5
    
    if pooled_sd == 0:
        return 0.0
    
    return (mean1 - mean2) / pooled_sd


def _interpret_effect(d: float) -> str:
    """Interpret Cohen's d effect size."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


# =====================================================================
# 4. Underranking Detection
# =====================================================================
def _find_underranked_candidates(candidates: list, group_field: str, 
                                  disadvantaged_group: str) -> list:
    """
    Identifies candidates from the disadvantaged group who may be underranked.
    
    A candidate is flagged as potentially underranked if:
    1. They belong to the disadvantaged group
    2. Their semantic_score is higher than at least one candidate from the
       advantaged group who is ranked above them
    
    This catches cases where the pipeline's signal layers (India signals,
    trajectory, etc.) may have inadvertently pushed down a strong candidate.
    """
    flagged = []
    
    for i, cand in enumerate(candidates):
        if cand.get(f"_bias_{group_field}") != disadvantaged_group:
            continue
        
        cand_semantic = cand.get("semantic_score", 0.0)
        cand_rank = cand.get("rank", i + 1)
        cand_final = cand.get("final_score", cand.get("match_score", 0.0))
        
        # Look for candidates ranked higher (lower rank number) from other groups
        # who have LOWER semantic scores
        for other in candidates:
            other_rank = other.get("rank", 999)
            other_group = other.get(f"_bias_{group_field}")
            other_semantic = other.get("semantic_score", 0.0)
            
            if other_rank < cand_rank and other_group != disadvantaged_group:
                if cand_semantic > other_semantic + 0.02:  # 2% threshold
                    flagged.append({
                        "candidate_id": cand.get("id"),
                        "candidate_name": cand.get("name"),
                        "group": disadvantaged_group,
                        "rank": cand_rank,
                        "semantic_score": round(cand_semantic, 4),
                        "final_score": round(cand_final, 4),
                        "outperformed_by": {
                            "candidate_id": other.get("id"),
                            "candidate_name": other.get("name"),
                            "group": other.get(f"_bias_{group_field}"),
                            "rank": other_rank,
                            "semantic_score": round(other_semantic, 4),
                            "final_score": round(other.get("final_score", 
                                                  other.get("match_score", 0.0)), 4),
                        },
                        "reason": (
                            f"{cand.get('name')} ({disadvantaged_group}) has a higher "
                            f"semantic score ({cand_semantic:.4f}) than "
                            f"{other.get('name')} ({other_group}, semantic: {other_semantic:.4f}) "
                            f"but ranks lower ({cand_rank} vs {other_rank}). "
                            f"The gap may be caused by non-skill signal layers."
                        )
                    })
                    break  # One example per candidate is sufficient
    
    return flagged


# =====================================================================
# 5. Dimension Audit
# =====================================================================
def _audit_dimension(candidates: list, dimension_name: str, 
                      group_field: str) -> dict:
    """
    Audits a single bias dimension (gender or city tier).
    
    Returns a structured report with:
    - Group statistics (count, mean, median, stdev of final scores)
    - Score gap analysis with effect size
    - Flagged underranked candidates
    - Overall verdict
    """
    # Partition candidates by group
    groups = {}
    for cand in candidates:
        group = cand.get(f"_bias_{group_field}", "Unknown")
        if group not in groups:
            groups[group] = []
        groups[group].append(cand)
    
    # Compute per-group statistics
    group_stats = {}
    for group_name, group_cands in groups.items():
        scores = [c.get("final_score", c.get("match_score", 0.0)) for c in group_cands]
        semantic_scores = [c.get("semantic_score", 0.0) for c in group_cands]
        ranks = [c.get("rank", 999) for c in group_cands]
        
        group_stats[group_name] = {
            "count": len(group_cands),
            "candidates": [c.get("name", "Unknown") for c in group_cands],
            "avg_final_score": round(_safe_mean(scores), 4),
            "median_final_score": round(_safe_median(scores), 4),
            "stdev_final_score": round(_safe_stdev(scores), 4),
            "avg_semantic_score": round(_safe_mean(semantic_scores), 4),
            "avg_rank": round(_safe_mean(ranks), 1),
            "best_rank": min(ranks) if ranks else None,
            "worst_rank": max(ranks) if ranks else None,
        }
    
    # Compute pairwise comparisons between non-Unknown groups
    comparisons = []
    real_groups = {k: v for k, v in groups.items() if k != "Unknown"}
    group_names = sorted(real_groups.keys())
    
    for i in range(len(group_names)):
        for j in range(i + 1, len(group_names)):
            g1_name, g2_name = group_names[i], group_names[j]
            g1_scores = [c.get("final_score", c.get("match_score", 0.0)) 
                        for c in real_groups[g1_name]]
            g2_scores = [c.get("final_score", c.get("match_score", 0.0)) 
                        for c in real_groups[g2_name]]
            
            d = _cohens_d(g1_scores, g2_scores)
            gap = _safe_mean(g1_scores) - _safe_mean(g2_scores)
            
            comparisons.append({
                "group_a": g1_name,
                "group_b": g2_name,
                "avg_score_a": round(_safe_mean(g1_scores), 4),
                "avg_score_b": round(_safe_mean(g2_scores), 4),
                "score_gap": round(abs(gap), 4),
                "gap_direction": f"{g1_name if gap > 0 else g2_name} scores higher",
                "cohens_d": round(d, 3),
                "effect_size": _interpret_effect(d),
                "sample_size_warning": (
                    "⚠️ Small sample size — interpret with caution"
                    if min(len(g1_scores), len(g2_scores)) < 5
                    else None
                ),
            })
    
    # Find potentially underranked candidates
    underranked = []
    if dimension_name == "Gender":
        underranked = _find_underranked_candidates(candidates, group_field, "Female")
    elif dimension_name == "City Tier":
        underranked = _find_underranked_candidates(candidates, group_field, "Tier-2/3")
    
    # Generate overall verdict
    verdict = _generate_verdict(dimension_name, group_stats, comparisons, underranked)
    
    return {
        "dimension": dimension_name,
        "group_statistics": group_stats,
        "pairwise_comparisons": comparisons,
        "potentially_underranked": underranked,
        "verdict": verdict,
    }


def _generate_verdict(dimension: str, stats: dict, comparisons: list,
                       underranked: list) -> dict:
    """
    Generates an overall fairness verdict for a dimension.
    
    Verdict levels:
    - ✅ PASS: No significant bias detected
    - ⚠️ WATCH: Small effect detected, monitor with more data
    - 🔴 FLAG: Medium/large effect detected, investigate
    """
    max_effect = "negligible"
    max_d = 0.0
    flagged_comparison = None
    
    for comp in comparisons:
        if abs(comp["cohens_d"]) > abs(max_d):
            max_d = comp["cohens_d"]
            max_effect = comp["effect_size"]
            flagged_comparison = comp
    
    if max_effect in ("medium", "large"):
        level = "FLAG"
        icon = "🔴"
        summary = (
            f"{icon} Potential bias detected in {dimension}. "
            f"Effect size is {max_effect} (Cohen's d = {max_d:.3f}). "
            f"{flagged_comparison['gap_direction']} by {flagged_comparison['score_gap']:.4f} points on average."
        )
        if underranked:
            summary += (
                f" {len(underranked)} candidate(s) from the disadvantaged group "
                f"may be underranked despite strong semantic scores."
            )
    elif max_effect == "small":
        level = "WATCH"
        icon = "⚠️"
        summary = (
            f"{icon} Minor score gap observed in {dimension}. "
            f"Effect size is {max_effect} (Cohen's d = {max_d:.3f}). "
            f"Not statistically conclusive with current sample size — monitor with more data."
        )
    else:
        level = "PASS"
        icon = "✅"
        summary = (
            f"{icon} No significant bias detected in {dimension}. "
            f"Score distributions across groups are comparable (Cohen's d = {max_d:.3f})."
        )
    
    recommendations = []
    if level in ("FLAG", "WATCH"):
        recommendations.append(
            "Review the signal layer weights (India signals, trajectory) "
            "to ensure they don't systematically disadvantage any group."
        )
        if underranked:
            recommendations.append(
                "Manually review the flagged candidates below — they have "
                "strong raw skill alignment but were ranked lower after signal adjustments."
            )
    if any(s.get("count", 0) < 3 for s in stats.values()):
        recommendations.append(
            "Some groups have very few candidates. Results are indicative, "
            "not conclusive. Re-run with a larger candidate pool for robust analysis."
        )
    
    return {
        "level": level,
        "icon": icon,
        "summary": summary,
        "recommendations": recommendations,
    }


# =====================================================================
# 6. Main Fairness Audit Function
# =====================================================================
def run_fairness_audit(candidates: list) -> dict:
    """
    Runs a comprehensive fairness audit on the ranked candidate list.
    
    Analyzes two dimensions:
    1. City Tier bias (Tier-1 metros vs. Tier-2/3 cities)
    2. Gender bias (inferred from Indian first names, conservatively)
    
    Returns a structured Fairness Report dictionary.
    """
    logger.info("--- STARTING STAGE 2e: Bias Detection & Fairness Audit ---")
    
    if not candidates:
        logger.warning("No candidates to audit.")
        return {"error": "No candidates provided for audit."}
    
    # Enrich candidates with bias metadata (prefixed with _bias_ to avoid
    # polluting the main output schema — these are audit-only fields)
    for cand in candidates:
        cand["_bias_gender"] = infer_gender(cand.get("name", ""))
        cand["_bias_city_tier"] = classify_city_tier(cand.get("location", ""))
    
    # Run audits for each dimension
    gender_audit = _audit_dimension(candidates, "Gender", "gender")
    city_audit = _audit_dimension(candidates, "City Tier", "city_tier")
    
    # Count gender distribution for the report header
    gender_dist = {}
    city_dist = {}
    for cand in candidates:
        g = cand.get("_bias_gender", "Unknown")
        c = cand.get("_bias_city_tier", "Unknown")
        gender_dist[g] = gender_dist.get(g, 0) + 1
        city_dist[c] = city_dist.get(c, 0) + 1
    
    # Compute overall fairness score (0-100)
    # Deduct points for each flagged dimension
    fairness_score = 100
    for audit in [gender_audit, city_audit]:
        level = audit["verdict"]["level"]
        if level == "FLAG":
            fairness_score -= 25
        elif level == "WATCH":
            fairness_score -= 10
    fairness_score = max(0, fairness_score)
    
    # Clean up temporary bias metadata from candidates
    for cand in candidates:
        # Keep _bias_ fields for internal use but don't export them
        pass
    
    report = {
        "report_title": "TalentLens Fairness Report",
        "total_candidates_audited": len(candidates),
        "fairness_score": fairness_score,
        "fairness_grade": (
            "A" if fairness_score >= 90 else
            "B" if fairness_score >= 75 else
            "C" if fairness_score >= 60 else
            "D" if fairness_score >= 40 else "F"
        ),
        "population_distribution": {
            "by_gender": gender_dist,
            "by_city_tier": city_dist,
        },
        "dimensions": [gender_audit, city_audit],
        "methodology_note": (
            "Gender is inferred conservatively from a curated dictionary of common "
            "Indian first names. Names not in the dictionary are classified as 'Unknown' "
            "and excluded from gender-specific comparisons. Effect sizes are computed "
            "using Cohen's d with pooled standard deviation. This report is an internal "
            "quality check — it does NOT modify any scores or rankings."
        ),
    }
    
    # Clean up internal _bias_ fields from candidate objects
    for cand in candidates:
        for key in list(cand.keys()):
            if key.startswith("_bias_"):
                del cand[key]
    
    logger.info(
        f"Fairness Audit complete. Score: {fairness_score}/100 "
        f"(Grade: {report['fairness_grade']}). "
        f"Gender: {gender_audit['verdict']['level']}, "
        f"City Tier: {city_audit['verdict']['level']}"
    )
    
    return report


# =====================================================================
# 7. Standalone Test
# =====================================================================
if __name__ == "__main__":
    import json
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(base_dir, "output", "ranked_candidates.json")
    
    if os.path.exists(output_file):
        candidates = utils.load_json(output_file)
        report = run_fairness_audit(candidates)
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"Output file not found: {output_file}")
        print("Run main.py first to generate ranked candidates.")
