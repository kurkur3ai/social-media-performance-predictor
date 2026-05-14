# Social Media Performance Predictor

> Predicts the engagement tier (**LOW / MEDIUM / HIGH**) of Indian beverage brand Instagram posts using a hybrid ML pipeline: deterministic structural features + LLM-extracted semantic features fed into an Optuna-tuned XGBoost classifier, with SHAP explainability, RAG-based post retrieval, and a local LLM for human-readable verdict generation.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset Construction](#2-dataset-construction)
3. [Feature Engineering](#3-feature-engineering)
4. [Model Training Pipeline](#4-model-training-pipeline)
5. [Prediction Pipeline Architecture](#5-prediction-pipeline-architecture)
6. [Application Stack](#6-application-stack)
7. [Key Design Decisions & Rationale](#7-key-design-decisions--rationale)
8. [Results](#8-results)
9. [Repository Structure](#9-repository-structure)
10. [Setup & Running](#10-setup--running)

---

## 1. Project Overview

**Task:** Given metadata about an Instagram post (brand, format, caption, image description, posting time, collaborators), predict whether the post will achieve LOW, MEDIUM, or HIGH engagement relative to the brand's historical performance on that format.

**Brands covered:** Coca-Cola India · Pepsi India · Red Bull India · Sprite India · Thums Up  
**Post types:** Reels · Static posts · Albums  
**Dataset size:** 378 Instagram posts

**Why this problem is hard:**
- Small dataset (378 posts → ~300 training) with three classes makes generalisation difficult.
- Engagement is a relative concept — a 5% ER is HIGH for a mega-brand but LOW for a micro-creator. Absolute thresholds would be meaningless across brands and formats.
- Semantic content quality (tone, celebrity presence, production value) cannot be captured by simple text statistics — it requires language understanding.

---

## 2. Dataset Construction

### 2.1 Raw Data Source

The raw dataset is `assignment-dataset.json` — a nested JSON file where each record contains:

```
record
└── data
    ├── id                  — unique post identifier
    ├── metadata_content    — caption, media_name (reel/post/album), duration, created_at, is_collaborated_post, collaborators
    ├── profile_stats       — username (brand), followers
    ├── engagements         — likes, views, comments, shares, engagement_rate
    └── media[]             — media files; thumbnail entries include a pre-computed AI image summary
```

The `engagement_rate` provided in the dataset is a pre-computed field. It is used **exclusively** as the target signal — it is never used as an input feature, preventing data leakage.

### 2.2 Target Label Construction

**Decision: within-brand, within-media-type percentile rank — not global thresholds.**

Each post is ranked by `engagement_rate` within its `(brand, media_name)` group. The rank percentile is then discretised into three equal thirds:

| Percentile | Tier   |
|-----------|--------|
| 0–33%     | LOW    |
| 33–67%    | MEDIUM |
| 67–100%   | HIGH   |

**Rationale:** A Red Bull reel and a Sprite static post have fundamentally different engagement norms. Using global thresholds would systematically mislabel posts from brands with structurally different follower bases or content styles. Within-group percentile ranking ensures each tier is equally represented per brand-format cell, making the classification problem well-posed.

**Sparse-bucket fallback:** Groups with fewer than 10 posts cannot have reliable within-group statistics. For these, posts are ranked within their `media_name` alone (collapsing the brand dimension).

### 2.3 Exploratory Analysis

`eda.ipynb` contains 13 analysis sections covering:
- Engagement metric distributions (log-scale, right-skewed)
- Brand and media type breakdowns
- Collab vs non-collab impact
- Reel duration vs engagement
- Time-of-day and day-of-week patterns
- Caption length and hashtag count correlations
- Full numeric correlation heatmap

Key findings that informed feature engineering:
- Reels dominate view counts but posts/albums can achieve higher ER relative to followers.
- Collab posts show notably higher median engagement.
- Posting hour and seasonality (IPL season: Mar–May, summer: Mar–Jun) show measurable signal.
- Caption length and hashtag count have weak but non-zero correlation with engagement.

---

## 3. Feature Engineering

The full feature vector has **59 features** — a combination of structural (deterministic) and semantic (LLM-extracted) signals.

### 3.1 Structural Features (25)

Computed deterministically from post metadata. No model required; fully reproducible.

| Feature group | Features | Description |
|--------------|---------|-------------|
| **Temporal** | `post_hour`, `post_dow`, `post_month`, `post_quarter`, `is_weekend` | When the post was published |
| **Seasonality** | `is_ipl_season`, `is_summer` | IPL cricket season (Mar–May) and summer (Mar–Jun) — high engagement periods for beverage brands |
| **Hour bucket** | `hour_bucket` | 0=night, 1=morning, 2=afternoon, 3=evening — coarser but cleaner than raw hour |
| **Collaborators** | `num_collabs`, `is_influencer_post`, `collab_count_capped` | Number of tagged collaborators (capped at 3 to reduce outlier influence) |
| **Caption** | `caption_len`, `word_count`, `hashtag_count`, `mention_count`, `line_breaks`, `emoji_count`, `emoji_density`, `has_url` | Structural caption signals |
| **Media** | `is_reel`, `is_album`, `is_post`, `duration`, `duration_bin` | Format identity + reel duration discretised into 4 bins |
| **Audience** | `followers`, `log_followers`, `follower_tier` | Brand scale; log-transformed to reduce skew; tiered for XGBoost splits |

### 3.2 Semantic Features (34, one-hot encoded from 10 LLM fields)

Extracted by a local LLM (llama3.2 via Ollama) from the post caption + pre-computed image description. The LLM is prompted with a strict JSON schema to ensure consistent, structured output.

| LLM field | Type | Values | XGBoost encoding |
|-----------|------|--------|-----------------|
| `tone` | enum | humorous / hype / emotional / informational / casual_relatable | 5 one-hot columns |
| `language` | enum | english / hindi / hinglish / other | 4 one-hot columns |
| `production_quality` | enum | polished / UGC / mixed | 3 one-hot columns |
| `product_prominence` | enum | hero / supporting / absent | 3 one-hot columns |
| `energy_level` | enum | high / medium / low | 3 one-hot columns |
| `cta_type` | enum | engage_comment / engage_tag / visit_link / buy / none | 5 one-hot columns |
| `celebrity_presence` | bool | true / false | 0/1 |
| `is_hinglish` | bool | true / false | 0/1 |
| `has_question` | bool | true / false | 0/1 |
| `content_theme` | free text | comma-separated themes | **excluded** (not encodable) |

**Why exclude `content_theme`?** Free-text fields cannot be reliably one-hot encoded — the LLM produces too many theme combinations. Including it would require embedding or NLP preprocessing that adds complexity without guaranteed gain on 378 samples. It is extracted and cached for human-readable output only.

**LLM validation and caching:**
- Output is validated against allowed enum values with a 3-tier correction pipeline: exact match → case-fold match → known synonym alias → fallback to safe default.
- Every LLM call is SHA-256 keyed and cached to `llm_cache.json`. Cached results are reused across all subsequent runs, making inference deterministic and fast.
- Two LLM backends are supported: Ollama (local, default) and Groq (cloud). Switch via `LLM_BACKEND` in Cell 1 of `train.ipynb`.

---

## 4. Model Training Pipeline

### 4.1 Train / Validation Split

5-fold stratified cross-validation (sklearn `StratifiedKFold`, `random_state=42`). Stratified on `tier_int` (0/1/2) to ensure each fold has the same class distribution. Each fold is saved as a parquet file under `folds/`.

**Why CV instead of a single hold-out?** With 378 total posts (~75 per fold), a single 80/20 split gives only ~75 test examples — too few for reliable macro-F1 estimation. CV gives 5 independent estimates across all posts.

### 4.2 Hyperparameter Search (Optuna)

200-trial TPE (Tree-structured Parzen Estimator) search with `MedianPruner` (20 startup trials) on all 5 folds.

**Search space:**

| Parameter | Range |
|-----------|-------|
| `n_estimators` | 100–700 |
| `max_depth` | 2–7 |
| `learning_rate` | 0.005–0.3 (log) |
| `subsample` | 0.5–1.0 |
| `colsample_bytree` | 0.4–1.0 |
| `colsample_bylevel` | 0.4–1.0 |
| `min_child_weight` | 1–10 |
| `reg_alpha` | 0.0001–10 (log) |
| `reg_lambda` | 0.1–5.0 |
| `gamma` | 0.0–1.0 |
| `max_delta_step` | 0–5 |
| `w_low`, `w_med`, `w_high` | 0.5–3.0 (class weights) |

**Key decision — class weights as hyperparameters:** Rather than fixing weights with `compute_sample_weight("balanced")`, the three class weights are included in the Optuna search. This allows the search to find the jointly optimal weight–architecture combination. Fixed "balanced" weights can over- or under-correct depending on the data distribution and model complexity.

**Best parameters found:**

```python
n_estimators=418, max_depth=6, learning_rate=0.01748,
subsample=0.6748, colsample_bytree=0.676, colsample_bylevel=0.850,
min_child_weight=5, reg_alpha=0.3916, reg_lambda=2.708,
gamma=0.758, max_delta_step=3
# Class weights: LOW=0.899, MEDIUM=1.393, HIGH=1.508
```

### 4.3 XGBoost Configuration

```python
objective   = "multi:softmax"   # 3-class, outputs class index
num_class   = 3
tree_method = "hist"            # histogram-based, ~3× faster with identical results
```

### 4.4 Model Artefacts

| File | Description | Use |
|------|-------------|-----|
| `model_cv_best.json` | Best fold (fold 4) model | **Production inference** |
| `model_full.json` | Retrained on 100% of data, same Optuna params | Reference only |
| `feature_cols.json` | Ordered list of 59 feature column names | Required for inference |

**Why use `model_cv_best.json` for production?** The best CV fold model preserves a held-out validation set (fold 4: 75 posts) that serves as a clean RAG retrieval pool anchor and honest evaluation set. Using the full-data model would eliminate this separation.

---

## 5. Prediction Pipeline Architecture

Each call to `POST /predict` executes 7 steps:

```
Request (brand, media_name, caption, img_summary, ...)
    │
    ├─ 1. Structural features   → 25 deterministic features from metadata
    │
    ├─ 2. LLM semantic features → Ollama llama3.2 call (or cache hit)
    │                              Returns 10 fields; validated and coerced
    │
    ├─ 3. Feature vector        → One-hot encode + align to feat_cols (59 features)
    │
    ├─ 4. XGBoost inference     → Predict class + class probabilities
    │
    ├─ 5. SHAP attribution      → Per-prediction feature importances
    │                              TreeExplainer with fallback to feature_importances_
    │
    ├─ 6. RAG retrieval         → Cosine similarity over 303 training posts
    │                              (75 held-out val posts excluded from pool)
    │                              Uses all-MiniLM-L6-v2 sentence embeddings
    │
    └─ 7. LLM verdict           → Ollama llama3.2 with full context:
                                   model output + SHAP signals + similar posts +
                                   ER benchmarks per tier × media type
                                   Returns {verdict, explanation} JSON
```

### 5.1 SHAP Attribution

`shap.TreeExplainer` is used for per-prediction attribution. XGBoost 3.x introduced a vector `base_score` format incompatible with some SHAP versions. A two-stage fallback handles this:

1. Try `TreeExplainer(_model, feature_perturbation="tree_path_dependent")`
2. Try `TreeExplainer(_model)` (default perturbation)
3. Fall back to global `model.feature_importances_` (gain-based, not per-prediction)

### 5.2 RAG Retrieval

The 303 training posts are embedded using `all-MiniLM-L6-v2` (384-dim) and stored in `history_embeddings.npy`. On each prediction, the query `caption + img_summary` is embedded and cosine-compared against all 303 posts. Posts above the similarity threshold are returned enriched with their actual tier and engagement rate.

**Why exclude the 75 validation posts from RAG?** Using validation posts as retrieval examples for predictions on those same posts would be test data leakage — the LLM explanation would be informed by ground-truth information it should not have access to during evaluation.

### 5.3 ER Reference Statistics

At startup, engagement rate statistics are computed from the 303 training posts at two granularities:

- **Tier-level:** median, Q25, Q75 per tier (LOW/MEDIUM/HIGH)
- **Media × tier:** median, Q25, Q75 per (media_name, tier) combination

These are injected into the LLM prompt as concrete benchmarks. Cells with n<15 are flagged as small-sample so the LLM hedges appropriately.

**Training-set ER benchmarks (reel format):**

| Tier | Median ER | IQR | n |
|------|-----------|-----|---|
| LOW | 0.50% | 0.07%–1.60% | 76 |
| MEDIUM | 2.15% | 1.41%–5.28% | 83 |
| HIGH | 7.21% | 3.44%–10.25% | 81 |

---

## 6. Application Stack

```
┌─────────────────────────────────────────┐
│  main.py                                │
│  Subprocess launcher — starts both      │
│  backend and frontend, validates        │
│  required artefacts on disk             │
└─────────────────────────────────────────┘
          │                    │
          ▼                    ▼
┌──────────────────┐  ┌──────────────────┐
│  backend.py      │  │  frontend.py     │
│  FastAPI         │  │  Streamlit       │
│  Port 8000       │  │  Port 8501       │
└──────────────────┘  └──────────────────┘
     │
     ├── GET  /health               — readiness probe
     ├── GET  /validation_examples  — 75 held-out posts for testing
     └── POST /predict              — full inference pipeline
```

**Frontend tabs:**
- **Validation Set** — browse and analyse the 75 held-out posts with correct-label comparison
- **New Post** — input any post details and get a live prediction

---

## 7. Key Design Decisions & Rationale

| Decision | Alternative considered | Rationale |
|----------|----------------------|-----------|
| 3-class classification (LOW/MEDIUM/HIGH) | Regression on ER | Classification is more actionable; percentile thresholds make the problem well-posed across brands |
| Within-group percentile labeling | Global ER thresholds | Brand/format norms differ significantly; global thresholds would systematically bias predictions |
| LLM semantic features | Manual text features (TF-IDF, sentiment) | Caption quality is highly contextual (Hinglish humour, celebrity callbacks, IPL references) — hard to capture with bag-of-words |
| SHA-256 LLM response cache | No cache / API-only | Makes training and inference reproducible and fast; avoids redundant API calls |
| Class weights as Optuna hyperparameters | Fixed "balanced" weights | Jointly optimises class balance correction with model structure |
| `tree_method="hist"` | `"exact"` | ~3× training speed with negligible accuracy difference on tabular data |
| `model_cv_best.json` for production | `model_full.json` | Preserves a clean held-out eval set for RAG exclusion and honest evaluation |
| RAG from training pool only | RAG from all 378 posts | Prevents test data leakage — val posts excluded from the embedding index |
| SHAP TreeExplainer | SHAP KernelExplainer | Model-native for XGBoost; orders of magnitude faster |
| Local Ollama (llama3.2) | GPT-4 / Groq cloud | No API costs or rate limits; Groq backend also supported via config switch |
| ER reference per media × tier | Per-tier only | More specific benchmark allows concrete comparison across formats |

---

## 8. Results

**5-Fold Stratified Cross-Validation (XGBoost, Optuna-tuned):**

| Metric | Score |
|--------|-------|
| CV Macro-F1 | **0.424 ± 0.028** |
| CV Weighted-F1 | ~0.47 |
| CV Accuracy | ~0.50 |

**Best fold (fold 4) classification report:**

```
              precision    recall  f1-score   support

         LOW       0.56      0.64      0.60        25
      MEDIUM       0.31      0.28      0.29        25
        HIGH       0.58      0.56      0.57        25

    accuracy                           0.49        75
   macro avg       0.48      0.49      0.49        75
```

MEDIUM is the hardest tier to predict — it sits between two more distinguishable extremes. LOW and HIGH are predicted with ~56–58% precision, significantly above the 33% random baseline. The macro-F1 of 0.424 represents a ~28% relative improvement over chance on a balanced 3-class problem.

---

## 9. Repository Structure

```
.
├── assignment-dataset.json       # Raw input data (378 posts)
├── eda.ipynb                     # Exploratory data analysis (13 sections)
├── train.ipynb                   # Full training pipeline (14 cells)
├── backend.py                    # FastAPI prediction service
├── frontend.py                   # Streamlit UI (validation + new post tabs)
├── main.py                       # Subprocess launcher for both services
├── requirements.txt              # Python dependencies
│
├── model_cv_best.json            # Trained XGBoost model (fold 4, production)
├── model_full.json               # Trained XGBoost model (full data, reference)
├── feature_cols.json             # Ordered list of 59 feature column names
├── dataset_final.parquet         # Processed feature matrix (378 × 62)
├── llm_cache.json                # Cached LLM semantic extractions (378 entries)
├── history_embeddings.npy        # Sentence embeddings for 303 training posts
│
└── folds/
    ├── fold_0_train.parquet  …   # Training splits for each fold
    └── fold_4_val.parquet        # Held-out validation set (75 posts)
```

---

## 10. Setup & Running

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally with `llama3.2` pulled:
  ```bash
  ollama pull llama3.2
  ```

### Installation

```bash
git clone https://github.com/kurkur3ai/social-media-performance-predictor.git
cd social-media-performance-predictor

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### Running the Training Pipeline

All training artefacts are committed and ready to use. To retrain from scratch:

```bash
jupyter notebook train.ipynb
# Run all cells in order (Cell 1 → Cell 14)
```

> LLM extraction (Cell 7) is skipped for cached posts — completes in seconds with `llm_cache.json` present. Without cache, allow 10–20 minutes depending on GPU.

### Running the Application

```bash
python main.py
```

Opens:
- **Frontend** → http://localhost:8501
- **API docs** → http://localhost:8000/docs
- **Health check** → http://localhost:8000/health

### Optional: Groq Cloud Backend

Add `GROQ_API_KEY=your_key` to a `.env` file, then set `LLM_BACKEND = "groq"` in `train.ipynb` Cell 1. The production backend always uses local Ollama.
```bash
cp .env.example .env
```

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Your Groq Cloud API key — get one at https://console.groq.com |

---

## Required Libraries

| Package | Used in | Purpose |
|---|---|---|
| `pandas` | eda, train | DataFrames |
| `numpy` | eda, train | Numerical operations |
| `matplotlib` | eda | Plotting |
| `seaborn` | eda | Statistical plots |
| `scikit-learn` | train | StratifiedKFold splits |
| `tqdm` | train | Progress bars |
| `pyarrow` | train | Parquet file I/O |
| `groq` | train | Groq Cloud LLM API |
| `python-dotenv` | train | Load `.env` secrets |
| `ipykernel` | both | Jupyter notebook kernel |

---

## Project Files

| File | Description |
|---|---|
| `assignment-dataset.json` | Raw Instagram post dataset (378 posts, 5 brands) |
| `eda.ipynb` | Exploratory data analysis |
| `train.ipynb` | Feature engineering + LLM extraction + dataset creation + folds |
| `llm_cache.json` | Auto-generated cache of LLM responses (keyed by SHA-256 hash) |
| `dataset_final.parquet` | Final feature matrix (auto-generated after running train.ipynb) |
| `folds/` | Stratified K-fold splits (auto-generated) |
