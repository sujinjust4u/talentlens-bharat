"""
TalentLens Stage 2d: Candidate Persona Clustering

Clusters candidates into 3-4 archetypes using unsupervised learning (KMeans)
on their profile embeddings. Labels clusters as:
  - "The Generalist Builder" (broad stack, startup-ready)
  - "The Domain Expert" (strong domain background, weaker on specific advanced techs)
  - "The Deep Specialist" (focused high depth skills)

Recommends the best cluster for the job and provides a structured recommendation.
"""

import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import utils

logger = utils.setup_logging()

def cluster_candidates(parsed_jd: dict, ranked_candidates: list) -> dict:
    """
    Groups candidates into clusters using KMeans on profile text embeddings.
    Returns a dictionary containing:
      - best_cluster: str
      - cluster_recommendation: str
      - candidate_clusters: dict mapping candidate_id to cluster label
      - cluster_groups: dict mapping cluster label to list of candidates
    """
    logger.info("Starting Stage 2d: Candidate Persona Clustering...")
    
    n_candidates = len(ranked_candidates)
    if n_candidates == 0:
        return {
            "best_cluster": "N/A",
            "cluster_recommendation": "No candidates available for clustering.",
            "candidate_clusters": {},
            "cluster_groups": {}
        }

    # 1. Determine optimal cluster count (K)
    n_clusters = min(3, n_candidates)
    
    # 2. Compute embeddings on the fly for KMeans
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import KMeans
        model = SentenceTransformer(model_name)
    except ImportError as e:
        logger.error(f"Failed to import SentenceTransformer or KMeans: {str(e)}")
        # Graceful fallback if sklearn or sentence-transformers is missing
        return {
            "best_cluster": "The Generalist Builder",
            "cluster_recommendation": "Clustering fallback active due to dependency issues.",
            "candidate_clusters": {c["id"]: "The Generalist Builder" for c in ranked_candidates},
            "cluster_groups": {"The Generalist Builder": ranked_candidates}
        }

    # Import build_candidate_profile_text locally
    import stage2_ranker
    cand_texts = [stage2_ranker.build_candidate_profile_text(c) for c in ranked_candidates]
    embeddings = model.encode(cand_texts, convert_to_numpy=True)

    # 3. Fit KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

    # 4. Group candidates by cluster index
    groups = {i: [] for i in range(n_clusters)}
    for idx, c in enumerate(ranked_candidates):
        groups[cluster_labels[idx]].append(c)

    # 5. Profile each cluster to assign human-readable labels
    target_domain = parsed_jd.get("domain", "").strip().lower()
    
    cluster_stats = []
    for cid in range(n_clusters):
        cands = groups[cid]
        
        # Calculate average number of skills
        total_skills = 0
        for c in cands:
            s = c.get("skills")
            if isinstance(s, list):
                total_skills += len(s)
        avg_skills = total_skills / len(cands) if cands else 0
        
        # Calculate domain match ratio
        domain_matches = 0
        for c in cands:
            c_domain = c.get("domain", "").strip().lower()
            if target_domain and target_domain in c_domain:
                domain_matches += 1
        domain_match_ratio = domain_matches / len(cands) if cands else 0
        
        # Calculate average final score
        avg_score = sum(c.get("final_score", 0.0) for c in cands) / len(cands) if cands else 0
        
        cluster_stats.append({
            "cid": cid,
            "avg_skills": avg_skills,
            "domain_match_ratio": domain_match_ratio,
            "avg_score": avg_score
        })

    # Assign labels based on profile metrics
    # Sort by skills count descending -> Generalist Builder is highest
    cluster_stats.sort(key=lambda x: x["avg_skills"], reverse=True)
    generalist_cid = cluster_stats[0]["cid"]
    
    # Remaining clusters
    remaining = [s for s in cluster_stats if s["cid"] != generalist_cid]
    
    domain_expert_cid = None
    deep_specialist_cid = None
    
    if len(remaining) > 0:
        # Sort remaining by domain match ratio descending -> Domain Expert
        remaining.sort(key=lambda x: x["domain_match_ratio"], reverse=True)
        domain_expert_cid = remaining[0]["cid"]
        
        if len(remaining) > 1:
            deep_specialist_cid = remaining[1]["cid"]

    # Build index to label map
    label_map = {}
    label_map[generalist_cid] = "The Generalist Builder"
    if domain_expert_cid is not None:
        label_map[domain_expert_cid] = "The Domain Expert"
    if deep_specialist_cid is not None:
        label_map[deep_specialist_cid] = "The Deep Specialist"

    # 6. Format output
    candidate_clusters = {}
    cluster_groups = {"The Generalist Builder": [], "The Domain Expert": [], "The Deep Specialist": []}
    
    for cid, cands in groups.items():
        label = label_map.get(cid, "The Generalist Builder")
        for c in cands:
            candidate_clusters[c["id"]] = label
            cluster_groups[label].append(c)

    # Clean empty lists from cluster groups
    cluster_groups = {k: v for k, v in cluster_groups.items() if len(v) > 0}

    # 7. Determine the "best" cluster (highest average final score)
    best_cid = max(cluster_stats, key=lambda x: x["avg_score"])["cid"]
    best_cluster = label_map.get(best_cid, "The Generalist Builder")

    # 8. Create recommendation text
    best_cands = cluster_groups[best_cluster]
    # Sort them by final_score descending
    best_cands.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
    top_5_names = [c["name"] for c in best_cands[:5]]
    
    rec_text = (
        f"For this role, {best_cluster} cluster is your best bet. "
        f"Here are {len(top_5_names)} from that cluster: {', '.join(top_5_names)}."
    )

    logger.info(f"Clustering complete. Best cluster: {best_cluster}")
    return {
        "best_cluster": best_cluster,
        "cluster_recommendation": rec_text,
        "candidate_clusters": candidate_clusters,
        "cluster_groups": cluster_groups
    }

if __name__ == "__main__":
    import json
    base_dir = os.path.dirname(os.path.abspath(__file__))
    jd_json = os.path.join(base_dir, "data", "parsed_jd.json")
    ranked_json = os.path.join(base_dir, "output", "ranked_candidates.json")
    
    if os.path.exists(jd_json) and os.path.exists(ranked_json):
        jd = utils.load_json(jd_json)
        cands = utils.load_json(ranked_json)
        res = cluster_candidates(jd, cands)
        print("\n=== CLUSTERING TEST SUCCESS ===")
        print(f"Best Cluster: {res['best_cluster']}")
        print(f"Recommendation: {res['cluster_recommendation']}")
        print("Groups:")
        for k, v in res["cluster_groups"].items():
            print(f"  {k}: {len(v)} candidates")
        print("===============================\n")
