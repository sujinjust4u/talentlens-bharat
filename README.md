# TalentLens Bharat 🇮🇳
### Intelligent Candidate Discovery — Redrob AI · India Runs Hackathon · Track 1

> Redrob AI ranks millions of profiles. Keyword filters miss the best ones.  
> TalentLens understands **why** a candidate fits — not just whether keywords match.

**Team:** Manu Krishnan (Lead) · Sujin S P (ML Engineer, Track 1) · OMPRAVEENKUMAR D · Livins LH  
**Track 1 built by:** Sujin S P (solo)

---

## Quickstart — reproduce submission in 4 steps

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Precompute candidate embeddings offline (run once)
python3 precompute.py --candidates candidates.jsonl.gz

# 3. Generate LLM reasoning cache offline (run once, needs GROQ_API_KEY)
python3 generate_reasoning.py --candidates candidates.jsonl.gz --top-n 150

# 4. Rank all 100K candidates and write submission CSV (zero network, ~6 seconds)
python3 rank.py --candidates candidates.jsonl.gz --out submission.csv

# Validate before submitting
python3 validate_submission.py submission.csv
```

Set your API key once:
```env
GROQ_API_KEY=gsk_your_key_here
```

---

## How it answers the JD

The job description explicitly says:

> *"The right answer involves reasoning about the gap between what the JD says and what the JD means."*

> *"A perfect-on-paper candidate who hasn't logged in for 6 months and has a 5% recruiter response rate is, for hiring purposes, not actually available."*

> *"That's a trap we've explicitly built into the dataset."*

TalentLens addresses all three directly:

| What the JD warns about | How TalentLens handles it |
|---|---|
| Keyword stuffers with no real experience | Semantic embedding ranks by meaning, not keywords. Expert skills with 0 duration months → honeypot cap (score 0.05) |
| Unavailable candidates who look great on paper | Behavioral signal layer: open_to_work + last_active_date + recruiter_response_rate + github_activity |
| Title-chasers hopping every 1.5 years | Explicit disqualifier: avg tenure <18 months over 3+ roles → ×0.80 penalty |
| Consulting-only careers (TCS, Infosys, Wipro) | Explicit disqualifier: consulting-only history → ×0.75 penalty |
| Offer ghosters | offer_acceptance_rate = 0.0 → ×0.85 penalty |
| Tier-2/3 hidden gems missed by keyword search | India signal layer: +0.15 city boost for 37+ Tier-2/3 cities |
| AI keywords without pre-LLM production experience | Career trajectory scoring: title progression speed + skill velocity rewards real experience depth |

---

## Scoring formula

```
Final Score = Disqualifier_Penalty × Stuffer_Penalty × (
    0.55 × Semantic_Score × Seniority_Multiplier
  + 0.20 × India_Signal_Score
  + 0.10 × Career_Trajectory_Score
  + 0.15 × Behavioral_Signal_Score
)
```

**Why these weights:**
- 55% semantic — core fit is the primary signal, not a tiebreaker
- 20% India signals — Redrob's mission is the Indian talent market; this operationalizes it
- 15% behavioral — availability matters as much as suitability for active hiring
- 10% trajectory — career momentum is a tiebreaker between similar semantic scores

---

## Pipeline architecture

```
OFFLINE (run once before submission):
  candidates.jsonl.gz ──► precompute.py ──► embeddings.npy (float16, ~73MB)
  job_description.txt ──► stage1_parser.py ──► parsed_jd.json
  top 150 candidates ──► generate_reasoning.py ──► reasoning_cache.json (150 LLM calls)

ONLINE RANKING (zero network, ~6 seconds):
  embeddings.npy ──► numpy batch dot product (cosine sim for all 100K simultaneously)
       │
       ▼
  honeypot filter ──► cap at 0.05 (605 caught)
       │
       ▼
  4-component blend + disqualifier penalties
       │
       ▼
  sort descending, ties broken by candidate_id ascending
       │
       ▼
  top 100 + reasoning from cache ──► submission.csv ✅
```

---

## Stage-by-stage breakdown

### Stage 1 — LLM job description parser (`stage1_parser.py`)

Converts raw JD text into a structured Pydantic JSON via `llama-3.1-8b-instant` (Groq).

Extracts:
- `required_skills` vs `nice_to_have` (explicitly separated — not the same list)
- `target_seniority`: intern / junior / mid / senior / lead / principal
- `target_domain`: Fintech, B2B SaaS, etc.
- `implicit_signals`: "scrappy product attitude", "ships before optimizing", "India market"

The structured profile — not the raw JD text — is what gets embedded. This means the semantic similarity is comparing candidate profiles against *what the role actually needs*, not whatever keywords the hiring manager happened to use.

---

### Stage 2 — Semantic ranker with seniority penalty (`stage2_ranker.py`)

Embeds all candidate profiles locally with `sentence-transformers/all-MiniLM-L6-v2` (no API cost, no rate limits, runs entirely offline). Computes cosine similarity via NumPy batch dot product — all 100K in a single pass.

**Seniority gap multiplier** — prevents a junior candidate with perfect keywords from outranking a senior:

| Gap from JD seniority | Multiplier |
|---|---|
| 0 — exact match | 1.00 |
| 1 level off | 0.92 |
| 2 levels off | 0.82 |
| 3+ levels off | 0.75 |

The JD says 5–9 years and targets senior level. A principal-level candidate still scores at 0.75× — overqualified is penalized just like underqualified.

---

### Stage 2b — India signal scoring (`stage2b_india_signals.py`)

Starting from baseline `0.50`:

| Signal | Adjustment |
|---|---|
| Tier-2/3 city (37+ cities: Madurai, Indore, Jaipur, Coimbatore, Nagpur…) | `+0.15` |
| Active in last 30 days | `+0.15` |
| Active in last 90 days | `+0.05` |
| Inactive > 180 days | `−0.05` |
| India-specific tech (UPI, Tally, Bhim, Rupay, Aadhaar, GSTIN, PhonePe, Razorpay…) | `+0.10` per match |
| Skills list null or empty | `−0.20` penalty + `recruiter_flag` |

The Tier-2/3 boost directly addresses what the JD's "final note for participants" highlights — hidden gems whose career history shows real product experience but whose profiles don't surface via keyword search.

---

### Stage 2c — Career trajectory scoring (`stage2c_trajectory.py`)

The JD explicitly says: *"Some people hit 'senior engineer' judgment at 4 years; some never hit it after 15."*

This stage measures actual career momentum, not years of experience:

- **Skill velocity** (up to 0.40): skills above current title's typical rank signal upward growth
- **Title progression speed** (up to 0.30): 2+ roles in <5 YOE = 0.30, <7 YOE = 0.20, 1+ role = 0.10
- **Advanced skill recency** (up to 0.30): senior/lead-tier skills acquired in last 12 months

Labels: `High momentum` / `Steady growth` / `Early stage` / `Plateau`

These labels feed Stage 3 — the LLM explainer adds trajectory language to justifications only when the signal is genuine, never for `Plateau` candidates.

---

### Stage 2c (ii) — Behavioral signal scoring

Reads directly from `redrob_signals` (the 23 platform behavioral fields):

| Signal | Contribution |
|---|---|
| `open_to_work_flag` = True | `+0.30` |
| `last_active_date` < 7 days | `+0.30` |
| `last_active_date` < 30 days | `+0.20` |
| `last_active_date` < 90 days | `+0.06` |
| `recruiter_response_rate` | Linear up to `+0.25` |
| `github_activity_score` > 0 | Up to `+0.15` |

The JD explicitly warns about unavailable candidates. This layer ensures a candidate with 6 months of inactivity and a 5% response rate — regardless of how strong their skills are — never reaches the top 100.

---

### Stage 2c (iii) — Disqualifier penalties

Multiplicative penalties applied to the blended score (minimum combined multiplier: 0.50):

| Disqualifier | Multiplier | JD reference |
|---|---|---|
| Title-chaser: avg tenure <18 months over 3+ roles | ×0.80 | "optimizing for Senior → Staff → Principal titles by switching companies every 1.5 years" |
| Consulting-only career (TCS, Infosys, Wipro, Accenture…) | ×0.75 | "people who have only worked at consulting firms" |
| Offer ghoster: acceptance rate = 0.0 | ×0.85 | behavioral availability |
| Unreliable: interview completion rate < 50% | ×0.90 | behavioral availability |
| Unresponsive: recruiter response rate < 20% | ×0.92 | behavioral availability |

---

### Stage 2c (iv) — Honeypot and keyword-stuffer defense

605 honeypots detected across 100K candidates. None entered the top 100.

**Hard detection** — score capped at 0.05:
- Expert or advanced proficiency with `duration_months = 0` (impossible)
- 10+ expert skills simultaneously

**Statistical penalty** — multiplicative penalties (floor 0.30):
- 12+ skills with zero total endorsements: ×0.50
- 8+ skills with zero endorsements: ×0.70
- 8+ expert/advanced skills but <5 total endorsements: ×0.65
- >6 skills per year of experience (density anomaly): ×0.85
- 3+ low assessment scores (<25) contradicting expert claims: ×0.80

---

### Stage 2d — Candidate persona clustering (`stage2d_clustering.py`)

KMeans unsupervised clustering on candidate profile embeddings. Groups the talent pool into 3–4 archetypes before the recruiter reads individual profiles:

- **The Deep Specialist** — narrow skills, high depth in one domain
- **The Generalist Builder** — broad stack, startup-ready (most relevant to this JD)
- **The Domain Expert** — strong industry background, domain-first

Recommends which cluster best fits the role based on cluster-average final scores.

---

### Stage 2e — Bias detection and fairness audit (`stage2e_bias_audit.py`)

The pipeline audits its own rankings. No enterprise ATS does this.

**What it checks:**
- **Gender audit**: Conservative inference from curated Indian first-name dictionary. Names not in the dictionary → `Unknown`, never guessed.
- **Geographic audit**: Tier-1 metros vs Tier-2/3 cities — verifies the India signal boost doesn't create excessive score gaps.

**Metrics computed:** group-level score distributions, Cohen's d effect size, underranked candidate detection (candidates whose semantic score exceeds a higher-ranked candidate from an advantaged group by >0.02).

**Verdicts:** `|d| < 0.20` = PASS · `0.20–0.50` = WATCH · `> 0.50` = FLAG

Output: Fairness Score 0–100, letter grade A–F, per-dimension verdicts, actionable recommendations.

---

### Stage 3 — LLM justifier and skill gap bridge (`stage3_explainer.py`)

For each top candidate, a structured LLM call generates three things:

**1. Trajectory-aware justification (2 sentences)**  
Trajectory language is added only when real (`High momentum` or `Steady growth`). For `Plateau` candidates it focuses on domain and seniority fit — no false optimism.

**2. Skill gap and learning bridge**  
Missing JD requirements, learning time estimated from trajectory label, 2–3 specific resources (Coursera links, GitHub repos, official docs) — not generic suggestions.

**3. Three targeted interview questions**  
Auto-generated to probe the candidate's specific identified gaps. Not generic technical questions — questions that only make sense for this candidate against this JD.

---

### Stage 4 — Reverse JD generator (`stage4_reverse_jd.py`)

Takes the top 3 ranked candidates and reconstructs what the ideal job description *should* say given the actual talent pool. Inverts the problem: instead of ranking candidates against the JD, it asks *"if these are your best matches, what role are you actually hiring for?"*

**Output:**
- Alignment score (0–100) comparing original JD to reconstructed ideal
- Skill comparison grid: Aligned / Missing in Candidates / Bonus in Candidates  
- Actionable JD rewrite suggestions

Example output for this JD:
```json
{
  "alignment_score": 95,
  "suggested_jd_rewrites": [
    "Replace Kafka with RabbitMQ — all top candidates familiar with RabbitMQ, none with Kafka.",
    "Add FastAPI as core requirement — present in top candidates, missing from JD.",
    "Remove financial auditing tools — missing across all top candidates."
  ]
}
```

---

## Production pipeline (`rank.py`) — technical specs

| Metric | Value |
|---|---|
| Dataset size | 100,000 candidates |
| Wall-clock runtime | 6.46 seconds |
| Peak memory usage | 415.7 MB |
| Memory limit | 16 GB |
| Network calls during ranking | 0 |
| GPU required | No |
| Honeypots detected | 605 |
| Honeypots in top 100 | 0 |
| LLM reasoning coverage | 100/100 |
| Fallback reasoning | 0 |
| Validator result | ✅ Submission is valid |

**Imports in `rank.py`:** `argparse`, `csv`, `gzip`, `json`, `os`, `sys`, `time`, `datetime`, `numpy` — zero network dependencies, verified by regex scan.

---

## Streamlit dashboard (`app.py`)

Live sandbox: [talentlens-bharat.streamlit.app](https://talentlens-bharat-p9mszttjf8x77caexawwvf.streamlit.app)

Two modes:

**Interactive Recruitment Explorer**  
Paste any JD → get ranked candidates with confidence scores, persona clusters, fairness report, skill gap bridges, interview questions, and reverse JD analysis.

**Production Submission Generator**  
Upload `candidates.jsonl.gz` → runs the full offline pipeline → validates → browser download of `submission.csv`.

---

## Project structure

```
talentlens-bharat/
├── rank.py                    # Main submission file — judges run this
├── precompute.py              # Offline: embed 100K candidates → embeddings.npy
├── generate_reasoning.py      # Offline: LLM reasoning for top 150 → reasoning_cache.json
├── stage1_parser.py           # LLM JD parser (LangChain + Groq)
├── stage2_ranker.py           # Semantic embeddings + seniority penalty
├── stage2b_india_signals.py   # India-specific signal scoring
├── stage2c_trajectory.py      # Career momentum scoring
├── stage2d_clustering.py      # KMeans persona clustering
├── stage2e_bias_audit.py      # Fairness audit (Cohen's d)
├── stage3_explainer.py        # LLM justifications + skill gap + interview Qs
├── stage4_reverse_jd.py       # Reverse JD generator
├── app.py                     # Streamlit dashboard
├── validate_submission.py     # Redrob's official format validator
├── submission_metadata.yaml   # Submission metadata (filled)
├── data/
│   ├── parsed_jd.json         # Pre-generated JD parse (ships with repo)
│   └── reasoning_cache.json   # Pre-generated reasoning for top 150
├── precomputed/               # embeddings.npy lives here (git-ignored, ~73MB)
└── requirements.txt
```

---

## What I'd build next with access to Redrob's production data

| Feature | Why |
|---|---|
| Supervise seniority model on Redrob hire/no-hire labels | Replace heuristic gap multipliers with a model trained on actual recruiter decisions |
| A/B test India signal layer on/off | Validate that 70/20/10 weighting actually improves shortlist acceptance rates |
| Hindi and Tamil JD parsing | Capture requirements written in regional languages — keyword extraction misses these entirely |
| Recruiter feedback loop | Online learning: thumbs-up/down on recommendations updates signal weights |
| Live Redrob profile API integration | Replace static JSONL with real-time talent pool queries |

**Honest uncertainty:** The India signal weights (+0.15 for Tier-2/3 cities, +0.10 per India-tech match) are heuristic. They need validation against real recruiter decisions to confirm they improve shortlist quality, not just shift rankings. This is the first thing I'd A/B test.

---

## Installation

```bash
git clone https://github.com/sujinjust4u/talentlens-bharat
cd talentlens-bharat
pip install -r requirements.txt

# Add API key
echo "GROQ_API_KEY=gsk_your_key_here" > .env
```

Requires Python 3.9+. No GPU. All embeddings run locally with `sentence-transformers/all-MiniLM-L6-v2`.

---

## Built by

**Sujin S P** — B.Tech AI & ML (3rd year), SRM Institute of Science and Technology, Trichy  
Track 1 solo build · TalentLens Bharat · India Runs Hackathon · Redrob AI × Hack2Skill · June 2026
