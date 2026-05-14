# Social Media Performance Predictor — Final System Design & Execution Plan

## Executive Summary

The goal of this project is to build a robust and explainable system for predicting Instagram post performance across multiple beverage brands.

After analyzing both the assignment requirements and the dataset characteristics, the final system design prioritizes:

* evaluation rigor,
* explainability,
* robustness under small-data constraints,
* practical implementation,
* and graceful handling of noisy engagement metrics.

Several approaches were intentionally rejected during the design phase:

* Deep multimodal architectures were rejected because the dataset is far too small for stable end-to-end training.
* Raw engagement regression was rejected because engagement metrics are highly inconsistent across brands and content formats.
* Per-brand models were rejected because some brands contain too few samples for reliable specialization.
* LLM-only prediction pipelines were rejected because they are difficult to evaluate rigorously and introduce nondeterministic behavior.

The final architecture combines:

* structured feature engineering,
* lightweight semantic extraction,
* similarity retrieval,
* and an interpretable gradient-boosted classifier.

The system is designed to maximize reliability and explainability rather than raw model complexity.

---

# 1. Problem Understanding

This is NOT a standard regression problem.

The dataset contains:

* multiple brands,
* different audience sizes,
* reels vs static posts,
* influencer collaborations,
* sports campaigns,
* meme content,
* festival content,
* highly skewed engagement metrics.

Raw metrics are not directly comparable.

Examples:

* static posts naturally have views = 0,
* reels can have 100M+ views,
* engagement rate varies wildly across brands and formats.

Therefore:
predicting raw views or raw likes directly is fundamentally unstable.

---

# 2. Final Modeling Objective

## Final Prediction Target

Instead of predicting:

* likes,
* views,
* raw engagement rate,

we predict:

* LOW
* MEDIUM
* HIGH
* VIRAL

based on percentile ranking within:

brand × content_type

---

# 3. Why Relative Tiers Are Correct

This solves multiple dataset problems simultaneously:

| Problem                   | Solution                              |
| ------------------------- | ------------------------------------- |
| Different follower counts | Percentile normalization              |
| Reels vs posts mismatch   | Normalize within content type         |
| Outlier viral posts       | Tiering stabilizes target             |
| Small dataset             | Classification easier than regression |
| Explainability            | Human-friendly outputs                |

---

# 4. Final Tier Definition

For each:

brand × content_type

compute percentile rank of engagement rate.

Then:

| Percentile | Tier   |
| ---------- | ------ |
| 0–25       | LOW    |
| 25–60      | MEDIUM |
| 60–85      | HIGH   |
| 85–100     | VIRAL  |

Fallback:
If bucket size < 10:

* use content_type-only normalization.

This is critical for sparse buckets.

---

# 5. Dataset Analysis

## Important Dataset Characteristics

### Small Dataset

The dataset contains approximately 375 posts across 5 brands.

Implications:

* deep learning is not appropriate,
* regression targets will be unstable,
* feature engineering matters more than model complexity,
* evaluation rigor becomes extremely important.

---

### Highly Inconsistent Engagement Metrics

The dataset contains:

* likes,
* comments,
* shares,
* views,
* engagement_rate.

However:

* reels naturally receive far higher views,
* static posts frequently have views = 0,
* collaborations distort engagement distribution,
* some campaigns create extreme viral outliers.

Therefore:
raw engagement metrics are not directly comparable.

---

### Strong Seasonal & Cultural Signals

Observed themes:

* IPL campaigns,
* cricket collaborations,
* summer campaigns,
* regional language content,
* meme-driven humor,
* influencer collaborations.

This suggests:
semantic context is likely one of the strongest predictive signals.

---

### Sparse Brand Buckets

Some brands contain very few examples for certain content types.

Example:

* albums for Sprite,
* static posts for ThumsUp.

This makes:

* per-brand models unreliable,
* aggressive specialization dangerous.

The architecture must gracefully fall back to global behavior.

---

# 6. Final Architecture Decision

## Final Architecture Decision

A single unified model architecture was selected.

### Rejected: Per-Brand Models

Per-brand models were initially considered because each brand has a distinct content strategy and engagement distribution.

However, this approach was rejected because:

* several brands contain fewer than 30 posts,
* some brand × content-type buckets contain extremely sparse data,
* separate models would overfit heavily,
* and cold-start behavior for new brands would be poor.

### Selected: Unified Global Model

A unified model allows:

* sharing signal across brands,
* learning global engagement patterns,
* more stable generalization,
* and simpler evaluation.

Brand identity is incorporated as a feature rather than separating the modeling pipeline.

---

---

# 7. Final System Architecture

INPUT
(caption + image/video + metadata)
↓
Visual Summary Extraction
↓
Feature Engineering
↓
Embedding + Similarity Retrieval
↓
Classifier
↓
Explanation Engine
↓
Prediction API Response

---

# 8. Visual Processing Strategy

The dataset already contains AI-generated visual summaries for most posts.

These summaries include:

* OCR text,
* scene descriptions,
* entity detection,
* brand visibility,
* and high-level semantic context.

### Rejected: End-to-End Vision Modeling

Several alternatives were considered:

* CLIP embeddings,
* multimodal transformers,
* raw image embeddings,
* and fine-tuned visual encoders.

These approaches were rejected because:

* the dataset is far too small,
* visual representations would overfit quickly,
* training would become unnecessarily complex,
* and the assignment already provides structured vision outputs.

### Final Decision

The provided summaries are treated as the primary visual signal source.

At inference time:

* uploaded images/videos are converted into structured summaries using a vision-capable LLM,
* matching the format used during training.

This keeps train and inference distributions aligned while avoiding unnecessary multimodal complexity.

---

---

# 9. Feature Engineering Plan

## A. Metadata Features

| Feature            | Why                           |
| ------------------ | ----------------------------- |
| media_type         | strongest engagement driver   |
| duration           | reels vary strongly by length |
| collaborator_count | audience amplification        |
| is_collaborated    | influencer lift               |
| posting month      | IPL / summer seasonality      |
| weekday            | posting behavior              |
| brand              | identity prior                |
| influencer_post    | different dynamics            |

---

## B. Caption/Text Features

| Feature           | Why                      |
| ----------------- | ------------------------ |
| caption length    | short-form trend         |
| hashtag count     | campaign indicator       |
| emoji density     | casual tone              |
| CTA presence      | boosts comments/shares   |
| question presence | engagement bait          |
| Hinglish score    | localized relatability   |
| language          | regional content effects |

---

## C. Semantic Features

Extracted using:

* LLM,
* rules,
* embedding similarity.

| Feature            | Example                     |
| ------------------ | --------------------------- |
| content theme      | cricket / comedy / food     |
| cultural relevance | IPL / FIFA / summer         |
| tone               | humorous / hype / emotional |
| celebrity presence | creators / athletes         |
| production quality | UGC vs polished             |
| product prominence | subtle vs hero              |
| energy level       | calm vs high-energy         |

---

# 10. Embedding Strategy

Sentence-transformer embeddings are used only for semantic retrieval and similarity search.

Selected model:

* all-MiniLM-L6-v2

### Rejected: Embeddings as Direct Predictive Features

Using high-dimensional embeddings directly inside the classifier was rejected because:

* the dataset is too small,
* embedding dimensions are large relative to sample count,
* and the model would likely memorize semantic clusters rather than generalize.

### Final Decision

Embeddings are used only for:

* similarity retrieval,
* semantic clustering,
* and explanation support.

The predictive pipeline itself remains feature-based and interpretable.

---

---

# 11. Similar Historical Post Retrieval

For every new post:
retrieve:

* top 5 semantically similar historical posts.

Purpose:

* improve explainability,
* provide qualitative grounding,
* increase trust in predictions.

Example:

“This resembles previous IPL collaboration reels from Sprite that historically performed in the HIGH tier.”

Important:
retrieval is used for explanation and qualitative comparison.
It should NOT directly override model predictions.

---

# 12. Final ML Model Choice

XGBoost was selected as the primary predictive model.

### Rejected: Deep Learning Models

The following approaches were considered and rejected:

* transformers,
* neural networks,
* multimodal deep learning,
* end-to-end vision-language systems.

These approaches were rejected because:

* the dataset is too small,
* model variance would be extremely high,
* training instability would increase,
* and evaluation would become less reliable.

### Rejected: Pure LLM Prediction

An LLM-only prediction pipeline was also rejected because:

* predictions would be nondeterministic,
* rigorous cross-validation would become difficult,
* and benchmarking against baselines would be less meaningful.

### Final Decision

XGBoost provides:

* strong performance on small structured datasets,
* robustness to missing values,
* fast inference,
* and interpretable feature contributions.

This makes it significantly more appropriate for the dataset scale and assignment constraints.

---

---

# 13. LLM Usage Strategy

LLMs are used only where semantic understanding is genuinely necessary.

### Final Usage Scope

| Stage                       | LLM Used? |
| --------------------------- | --------- |
| vision summary generation   | yes       |
| semantic feature extraction | yes       |
| explanation generation      | optional  |
| prediction generation       | no        |

### Rejected: Heavy LLM Orchestration

Several more complex designs were intentionally rejected:

* LLM-generated predictions,
* agentic pipelines,
* multi-stage reasoning chains,
* and LLM-based scoring systems.

These approaches were rejected because:

* outputs become difficult to evaluate rigorously,
* reproducibility decreases,
* debugging becomes significantly harder,
* and prediction consistency suffers.

### Final Decision

The predictive component remains:

* deterministic,
* reproducible,
* and fully measurable using standard ML evaluation.

LLMs are restricted to semantic extraction and explanation layers only.

---

---

# 14. Explainability Strategy

## A. SHAP

Use:

* feature contribution explanations.

---

## B. Similar Historical Posts

Show:

* nearest successful posts,
* historical comparisons.

---

## C. Human Explanation Layer

Generate:

* concise reasoning summaries.

Example:

Predicted HIGH because:

* IPL-related content historically performs well,
* reel format outperforms static posts,
* celebrity collaboration boosts share probability,
* humor-heavy Hinglish captions align with brand behavior.

---

# 15. Evaluation Strategy

This assignment heavily prioritizes evaluation rigor.

This section matters more than fancy modeling.

---

## A. Stratified Cross Validation

Use:

* StratifiedGroupKFold

Stratify by:

* tier,
* brand.

Avoid leakage.

---

## B. Leave-One-Brand-Out Evaluation

Train:

* 4 brands

Test:

* unseen 5th brand.

This demonstrates:

* generalization,
* cold-start robustness.

---

## C. Baselines

We MUST compare against:

| Baseline          | Purpose                    |
| ----------------- | -------------------------- |
| random            | floor                      |
| always-medium     | sanity                     |
| brand median      | weak heuristic             |
| metadata-only XGB | value of semantic features |

Without baselines:
evaluation is weak.

---

# 16. Metrics

## Primary Metrics

* Macro F1
* Balanced Accuracy
* Confusion Matrix
* Per-brand F1
* Calibration score

NOT:

* plain accuracy.

---

# 17. Failure Analysis Plan

Analyze:

* viral outliers,
* creator posts,
* sparse brand buckets,
* static post anomalies,
* seasonal spikes.

Show:

* where model fails,
* why it fails,
* what additional data would solve it.

This demonstrates maturity.

---

# 18. Edge Case Handling

| Edge Case     | Handling               |
| ------------- | ---------------------- |
| views = 0     | treat as static post   |
| missing media | degrade gracefully     |
| expired S3    | skip visual features   |
| unseen brand  | use global priors      |
| no caption    | visual-only features   |
| no image      | text-only features     |
| sparse brand  | fallback normalization |

---

# 19. API Design

## Backend

FastAPI

### Endpoints

* POST /predict
* POST /analyze
* GET /health

---

# 20. Frontend Recommendation

## Recommended

Streamlit

Reason:

* fastest iteration,
* enough for assignment,
* easy demos,
* easy file uploads,
* easy visual explanation display.

Do NOT waste time on frontend polish.

---

# 21. Final Recommended Tech Stack

| Layer          | Stack                 |
| -------------- | --------------------- |
| Backend        | FastAPI               |
| ML             | XGBoost               |
| Embeddings     | sentence-transformers |
| Explainability | SHAP                  |
| Data           | pandas + sklearn      |
| Frontend       | Streamlit             |
| Vision         | LLM vision API        |
| Deployment     | local/docker          |

---

# 22. Final Design Principles

The final system design prioritizes:

* deterministic behavior,
* reproducibility,
* evaluation rigor,
* interpretability,
* and implementation simplicity.

Several ideas were intentionally rejected during planning:

* Deep multimodal architectures were rejected because they are not justified by dataset scale.
* Large feature sets were rejected because they increase overfitting risk without proportional signal gain.
* Complex ensembles were rejected because they add evaluation complexity while providing limited practical value.
* Retrieval-weighted prediction blending was rejected because it makes the predictive behavior harder to reason about.

The final architecture intentionally remains:

* compact,
* explainable,
* and evaluation-focused.

The goal is not maximizing architectural complexity.
The goal is building the most defensible and reliable system possible under the dataset constraints.

---

---

# 23. What Will Differentiate This Submission

The strongest differentiators will be:

1. rigorous evaluation,
2. intelligent normalization strategy,
3. clean explainability,
4. similarity retrieval,
5. honest failure analysis,
6. graceful handling of sparse brands.

Not:

* fancy deep learning,
* complicated multimodal architectures.

---

# 24. Final Recommended Development Order

## Phase 1

* EDA
* target construction
* normalization

## Phase 2

* deterministic features
* baseline models

## Phase 3

* semantic features
* embeddings

## Phase 4

* explainability
* retrieval

## Phase 5

* API
* frontend

## Phase 6

* evaluation report
* failure analysis
* Loom demo

---

# 25. Final Recommendation

The strongest submission here is NOT:

* the most complex,
* the most “AI-looking”,
* the deepest model.

The strongest submission is:

* scientifically honest,
* rigorously evaluated,
* architecturally justified,
* interpretable,
* robust under small data constraints.

The final architecture should optimize for exactly that.
