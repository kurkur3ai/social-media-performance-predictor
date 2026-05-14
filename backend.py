"""
backend.py — FastAPI prediction service
Run standalone: uvicorn backend:app --reload --port 8000
Or via main.py which launches both backend and frontend.
"""

from __future__ import annotations

import hashlib
import json
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
MODEL_PATH      = BASE_DIR / "model_cv_best.json"   # best CV fold (fold 4), used for evaluation
FEAT_COLS_PATH  = BASE_DIR / "feature_cols.json"
DATA_PATH       = BASE_DIR / "assignment-dataset.json"
CACHE_PATH      = BASE_DIR / "llm_cache.json"
EMBED_CACHE_PATH = BASE_DIR / "history_embeddings.npy"
DATASET_FINAL_PATH = BASE_DIR / "dataset_final.parquet"
FOLD_VAL_PATH   = BASE_DIR / "folds" / "fold_4_val.parquet"  # held-out val of best fold

# ── Constants ─────────────────────────────────────────────────────────────────
TIER_LABELS     = ["LOW", "MEDIUM", "HIGH"]
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL    = "llama3.2"
EMBED_MODEL_ID  = "all-MiniLM-L6-v2"

KNOWN_BRANDS    = ["cocacola_india", "pepsiindia", "redbullindia", "sprite_india", "thumsupofficial"]
KNOWN_MEDIA     = ["album", "post", "reel"]

_EMOJI_PAT = re.compile(
    r"[\U0001F300-\U0001F9FF\U00002600-\U000027BF\U0001FA00-\U0001FA9F]",
    re.UNICODE,
)

LLM_SYSTEM_PROMPT = """\
You are a social media content analyst for Indian beverage brand Instagram posts.
Analyse the CAPTION and IMAGE DESCRIPTION, then output EXACTLY this JSON object.
No markdown fences, no extra keys, no explanation — start with { and end with }:

{"content_theme":"...","tone":"...","language":"...","is_hinglish":...,"cta_type":"...","has_question":...,"celebrity_presence":...,"production_quality":"...","product_prominence":"...","energy_level":"..."}

  tone             : humorous | hype | emotional | informational | casual_relatable
  language         : english | hindi | hinglish | other
  is_hinglish      : true | false
  cta_type         : engage_comment | engage_tag | visit_link | buy | none
  has_question     : true | false
  celebrity_presence : true | false
  production_quality : polished | UGC | mixed
  product_prominence : hero | supporting | absent
  energy_level     : high | medium | low"""

FALLBACK_SEMANTIC = {
    "content_theme": "", "tone": "casual_relatable", "language": "english",
    "is_hinglish": False, "cta_type": "none", "has_question": False,
    "celebrity_presence": False, "production_quality": "mixed",
    "product_prominence": "supporting", "energy_level": "medium",
}

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Social Media Performance Predictor", version="1.0.0")

# ── Lazy-loaded singletons ────────────────────────────────────────────────────
_model:               xgb.XGBClassifier | None = None
_feat_cols:           list[str] | None          = None
_explainer:           shap.TreeExplainer | None  = None
_embed_model:         SentenceTransformer | None = None
_history_df:          pd.DataFrame | None        = None
_history_embeddings:  np.ndarray | None          = None
_llm_cache:           dict                       = {}
_ollama_client:       OpenAI | None              = None
_er_stats:            dict                       = {}  # {tier: {median, q25, q75, n}}
_er_media_stats:      dict                       = {}  # {(tier, media_name): {median, q25, q75, n}}


# ── Singleton loaders ─────────────────────────────────────────────────────────

def _get_model() -> tuple[xgb.XGBClassifier, list[str], shap.TreeExplainer]:
    global _model, _feat_cols, _explainer
    if _model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(f"Model not found at {MODEL_PATH}. Run train.ipynb first.")
        _model = xgb.XGBClassifier()
        _model.load_model(str(MODEL_PATH))
        _feat_cols = json.loads(FEAT_COLS_PATH.read_text())
        try:
            _explainer = shap.TreeExplainer(_model, feature_perturbation="tree_path_dependent")
        except Exception:
            try:
                _explainer = shap.TreeExplainer(_model)
            except Exception:
                _explainer = None  # will fall back to feature_importances_
    return _model, _feat_cols, _explainer


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL_ID)
    return _embed_model


def _get_history() -> tuple[pd.DataFrame, np.ndarray]:
    global _history_df, _history_embeddings
    if _history_df is None:
        # Identify held-out validation post IDs — exclude from RAG pool
        val_post_ids: set[str] = set()
        if FOLD_VAL_PATH.exists() and DATASET_FINAL_PATH.exists():
            _vdf  = pd.read_parquet(str(FOLD_VAL_PATH))
            _full = pd.read_parquet(str(DATASET_FINAL_PATH))
            val_post_ids = set(_full.loc[_vdf.index, "post_id"].astype(str))

        with open(DATA_PATH) as f:
            raw = json.load(f)
        rows = []
        for item in raw:
            d  = item["data"]
            mc = d.get("metadata_content", {})
            ps = d.get("profile_stats", {})
            en = d.get("engagements", {})
            post_id = d.get("id", "")
            if post_id in val_post_ids:
                continue  # keep RAG pool clean — no test data leakage
            img_summary = ""
            for m in d.get("media", []):
                if m.get("type") == "thumbnail" and m.get("summary"):
                    img_summary = m["summary"]
                    break
            rows.append({
                "post_id":        post_id,
                "brand":          ps.get("username", ""),
                "media_name":     mc.get("media_name", ""),
                "caption":        mc.get("caption", ""),
                "img_summary":    img_summary,
                "engagement_rate": en.get("engagement_rate", 0.0),
                "likes":          en.get("likes", 0),
                "views":          en.get("views", 0),
                "created_at":     mc.get("created_at", ""),
            })
        _history_df = pd.DataFrame(rows)

        # Attach tier labels from dataset_final (training posts only)
        if DATASET_FINAL_PATH.exists():
            _full = pd.read_parquet(str(DATASET_FINAL_PATH))
            tier_map = dict(zip(_full["post_id"].astype(str), _full["tier"].astype(str)))
            _history_df["tier"] = _history_df["post_id"].map(tier_map).fillna("UNKNOWN")
        else:
            _history_df["tier"] = "UNKNOWN"
        embed_model = _get_embed_model()
        if EMBED_CACHE_PATH.exists():
            _cached = np.load(str(EMBED_CACHE_PATH))
            if len(_cached) == len(_history_df):
                _history_embeddings = _cached
        if _history_embeddings is None:
            texts = (
                _history_df["caption"].fillna("") + " " +
                _history_df["img_summary"].fillna("")
            ).tolist()
            _history_embeddings = embed_model.encode(
                texts, show_progress_bar=False, normalize_embeddings=True, batch_size=32
            )
            np.save(str(EMBED_CACHE_PATH), _history_embeddings)

    return _history_df, _history_embeddings


def _get_ollama() -> OpenAI:
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
    return _ollama_client


def _load_llm_cache() -> None:
    global _llm_cache
    if CACHE_PATH.exists():
        _llm_cache = json.loads(CACHE_PATH.read_text())


def _save_llm_cache() -> None:
    tmp = CACHE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(_llm_cache, ensure_ascii=False))
    tmp.replace(CACHE_PATH)


def _cache_key(caption: str, img_summary: str) -> str:
    payload = f"{OLLAMA_MODEL}||{caption}||{img_summary}"
    return hashlib.sha256(payload.encode()).hexdigest()


# ── Feature engineering ───────────────────────────────────────────────────────

def _build_structural(
    brand: str, media_name: str, duration: float, caption: str,
    followers: int, is_collab: bool, collaborators: list[str],
    created_at_str: str,
) -> dict:
    try:
        dt = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    except Exception:
        dt = datetime.now(timezone.utc)

    cap        = caption or ""
    num_collab = len(collaborators)
    words      = cap.split()
    word_count = max(len(words), 1)
    emoji_cnt  = len(_EMOJI_PAT.findall(cap))
    h          = dt.hour
    hour_bucket = (
        1 if 6 <= h <= 11 else
        2 if 12 <= h <= 17 else
        3 if 18 <= h <= 23 else 0
    )
    dur = float(duration)

    return {
        "followers":           followers,
        "duration":            dur,
        "is_collab":           int(is_collab),
        "post_hour":           h,
        "post_dow":            dt.weekday(),
        "post_month":          dt.month,
        "post_quarter":        (dt.month - 1) // 3 + 1,
        "is_weekend":          int(dt.weekday() >= 5),
        "is_ipl_season":       int(dt.month in [3, 4, 5]),
        "is_summer":           int(dt.month in [3, 4, 5, 6]),
        "hour_bucket":         hour_bucket,
        "num_collabs":         num_collab,
        "is_influencer_post":  int(num_collab > 0),
        "collab_count_capped": min(num_collab, 3),
        "caption_len":         len(cap),
        "word_count":          len(words),
        "hashtag_count":       cap.count("#"),
        "mention_count":       cap.count("@"),
        "line_breaks":         cap.count("\n"),
        "emoji_count":         emoji_cnt,
        "emoji_density":       round(emoji_cnt / word_count, 4),
        "has_url":             int(bool(re.search(r"http|bit\.ly|link", cap, re.IGNORECASE))),
        "is_reel":             int(media_name == "reel"),
        "is_album":            int(media_name == "album"),
        "is_post":             int(media_name == "post"),
        "duration_bin":        (
            0 if dur == 0 else
            1 if dur <= 15 else
            2 if dur <= 30 else 3
        ),
        "log_followers":       float(np.log1p(followers)),
        "follower_tier":       (
            0 if followers < 50_000 else
            1 if followers < 200_000 else 2
        ),
    }


def _extract_semantic(caption: str, img_summary: str) -> dict:
    key = _cache_key(caption, img_summary)
    if key in _llm_cache:
        return _llm_cache[key]

    try:
        client   = _get_ollama()
        user_msg = (
            f"CAPTION:\n{caption.strip() or '(empty)'}\n\n"
            f"IMAGE DESCRIPTION:\n{img_summary.strip() or '(not available)'}"
        )
        resp = client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=512,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        result = json.loads(resp.choices[0].message.content)
        # Coerce booleans
        for bf in ("is_hinglish", "has_question", "celebrity_presence"):
            v = result.get(bf)
            if isinstance(v, str):
                result[bf] = v.strip().lower() in ("true", "1", "yes")
        _llm_cache[key] = result
        _save_llm_cache()
        return result
    except Exception:
        return dict(FALLBACK_SEMANTIC)


def _build_feature_vector(
    structural: dict,
    semantic: dict,
    brand: str,
    media_name: str,
    feat_cols: list[str],
) -> np.ndarray:
    row: dict = dict(structural)

    # ── One-hot: tone ─────────────────────────────────────────────────────────
    for v in ["casual_relatable", "emotional", "humorous", "hype", "informational"]:
        row[f"tone_{v}"] = int(semantic.get("tone") == v)

    # ── One-hot: language ─────────────────────────────────────────────────────
    for v in ["english", "hindi", "hinglish", "other"]:
        row[f"language_{v}"] = int(semantic.get("language") == v)

    # ── One-hot: production_quality ───────────────────────────────────────────
    for v in ["UGC", "mixed", "polished"]:
        row[f"production_quality_{v}"] = int(semantic.get("production_quality") == v)

    # ── One-hot: product_prominence ───────────────────────────────────────────
    for v in ["absent", "hero", "supporting"]:
        row[f"product_prominence_{v}"] = int(semantic.get("product_prominence") == v)

    # ── One-hot: energy_level ─────────────────────────────────────────────────
    for v in ["high", "low", "medium"]:
        row[f"energy_level_{v}"] = int(semantic.get("energy_level") == v)

    # ── One-hot: cta_type ─────────────────────────────────────────────────────
    for v in ["buy", "engage_comment", "engage_tag", "none", "visit_link"]:
        row[f"cta_type_{v}"] = int(semantic.get("cta_type") == v)

    # ── Boolean semantic ──────────────────────────────────────────────────────
    row["celebrity_presence"] = int(bool(semantic.get("celebrity_presence", False)))
    row["is_hinglish"]        = int(bool(semantic.get("is_hinglish", False)))
    row["has_question"]       = int(bool(semantic.get("has_question", False)))

    # ── One-hot: brand ────────────────────────────────────────────────────────
    for b in KNOWN_BRANDS:
        row[f"brand_{b}"] = int(brand == b)

    # ── One-hot: media_name ───────────────────────────────────────────────────
    for mn in KNOWN_MEDIA:
        row[f"media_name_{mn}"] = int(media_name == mn)

    # Align to training feature order; fill unseen columns with 0
    df = pd.DataFrame([row])
    for col in feat_cols:
        if col not in df.columns:
            df[col] = 0
    return df[feat_cols].values.astype(float)


# ── RAG retrieval ─────────────────────────────────────────────────────────────

def _retrieve_similar(
    caption: str,
    img_summary: str,
    top_k: int,
    threshold: float,
) -> list[dict]:
    query = f"{caption} {img_summary}".strip()
    if not query:
        return []

    hist_df, hist_emb = _get_history()
    embed_model       = _get_embed_model()
    q_emb             = embed_model.encode(
        [query], normalize_embeddings=True, show_progress_bar=False
    )
    sims = cosine_similarity(q_emb, hist_emb)[0]

    results = []
    for idx in np.argsort(sims)[::-1]:
        sim = float(sims[idx])
        if sim < threshold:
            break
        row = hist_df.iloc[idx]
        cap_text = str(row["caption"])
        results.append({
            "post_id":         str(row["post_id"]),
            "brand":           str(row["brand"]),
            "media_name":      str(row["media_name"]),            "tier":            str(row.get("tier", "UNKNOWN")),            "caption_snippet": cap_text[:120] + ("…" if len(cap_text) > 120 else ""),
            "similarity":      round(sim, 3),
            "engagement_rate": round(float(row["engagement_rate"]), 4),
            "created_at":      str(row.get("created_at", "")),
        })
        if len(results) >= top_k:
            break
    return results


# ── LLM explanation ───────────────────────────────────────────────────────────

def _generate_explanation(
    tier: str,
    proba: dict[str, float],
    top_features: list[dict],
    similar_posts: list[dict],
    semantic: dict,
    brand: str,
    media_name: str,
    num_collabs: int,
) -> dict[str, str]:
    feat_lines = "\n".join(
        f"  {i+1}. {f['feature']}: {f['shap_value']:+.3f}"
        for i, f in enumerate(top_features[:6])
    )
    sim_lines = (
        "\n".join(
            f"  - {p['brand']} {p['media_name']} | Tier={p.get('tier','?')} | ER={p['engagement_rate']:.2f} | sim={p['similarity']:.2f}"
            for p in similar_posts[:3]
        )
        or "  None found above threshold."
    )
    confidence = proba.get(tier, 0.0)
    low_p  = proba.get("LOW",    0.0)
    med_p  = proba.get("MEDIUM", 0.0)
    high_p = proba.get("HIGH",   0.0)
    # ER reference ranges — media-specific primary, tier-only fallback
    mn_key = media_name.lower().strip()
    def _er_line(t: str, mn: str) -> str:
        ms = _er_media_stats.get((t, mn), {})
        ts = _er_stats.get(t, {})
        if ms:
            caveat = " ⚠ small sample" if ms["n"] < 15 else ""
            return (f"{t} {mn}: median {ms['median']:.2f}% "
                    f"(IQR {ms['q25']:.2f}%–{ms['q75']:.2f}%, n={ms['n']}{caveat})")
        elif ts:
            return f"{t} (all formats): median {ts['median']:.2f}% (IQR {ts['q25']:.2f}%–{ts['q75']:.2f}%, n={ts['n']})"
        return f"{t}: no data"
    # Primary: specific media_name × tier row; secondary: all three tiers for the input media type
    primary_ms  = _er_media_stats.get((tier, mn_key), {})
    primary_ts  = _er_stats.get(tier, {})
    all_tier_lines = "\n  ".join(_er_line(t, mn_key) for t in ["LOW", "MEDIUM", "HIGH"])
    er_context = f"  {all_tier_lines}"
    prompt = (
        f"You are a senior social media analyst classifying an Instagram post's performance tier.\n\n"
        f"MODEL OUTPUT:\n"
        f"  Predicted tier: {tier} (confidence {confidence:.0%})\n"
        f"  Probabilities — LOW: {low_p:.0%} | MEDIUM: {med_p:.0%} | HIGH: {high_p:.0%}\n\n"
        f"POST CONTEXT:\n"
        f"  Brand: {brand} | Format: {media_name} | Tone: {semantic.get('tone')} "
        f"| Energy: {semantic.get('energy_level')} | Celebrity: {semantic.get('celebrity_presence')} "
        f"| Production: {semantic.get('production_quality')} | Theme: {semantic.get('content_theme') or 'general'}\n\n"
        f"KEY SHAP SIGNALS (positive = pushes toward {tier}, negative = pushes away):\n{feat_lines}\n\n"
        f"SIMILAR PAST POSTS (brand, media type, their tier, and their actual ER):\n{sim_lines}\n\n"
        f"ER BENCHMARKS for '{mn_key}' posts across tiers (training set):\n{er_context}\n"
        + (
            f"  → For this {mn_key}, {tier} peers: median {primary_ms['median']:.2f}% "
            f"(IQR {primary_ms['q25']:.2f}%–{primary_ms['q75']:.2f}%, n={primary_ms['n']})"
            + (" — treat as indicative only due to small n" if primary_ms["n"] < 15 else "")
            if primary_ms else
            (f"  → Tier-level benchmark: {tier} median {primary_ts['median']:.2f}% "
             f"(IQR {primary_ts['q25']:.2f}%–{primary_ts['q75']:.2f}%, n={primary_ts['n']})" if primary_ts else "")
        ) + "\n\n"
        f"Return ONLY valid JSON with exactly these two keys:\n"
        f'  "verdict": A 5-9 word phrase. Must name the tier, the media type ({media_name}), and either ER or a key signal. '
        f'Example: "HIGH reel — strong engagement, celebrity-backed". No full sentence.\n'
        f'  "explanation": Exactly 2 sentences. '
        f'Sentence 1: state which specific content signals (name at least 2 SHAP features) and the post format ({media_name}) '
        f'land it in {tier} rather than adjacent tiers — quote the actual ER benchmark for {tier} {mn_key} posts. '
        f'Sentence 2: reference the similar past posts by name, their tier, and their actual ER values; '
        f'compare whether the {media_name} format of those posts aligns with or differs from this post\'s classification. '
        f'Be specific with numbers. No improvement advice. No markdown. No extra keys.'
    )
    try:
        client = _get_ollama()
        resp   = client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=280,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(resp.choices[0].message.content.strip())
        verdict     = str(parsed.get("verdict",     "")).strip()
        explanation = str(parsed.get("explanation", "")).strip()
        if not verdict:
            verdict = f"{tier} — model confidence {confidence:.0%}"
        return {"verdict": verdict, "explanation": explanation}
    except Exception:
        confidence = proba.get(tier, 0.0)
        er = _er_stats.get(tier, {})
        er_note = f" typical ER for {tier} is {er['median']:.2f}%" if er else ""
        return {
            "verdict": f"{tier} {media_name} — {confidence:.0%} confidence",
            "explanation": (
                f"This {media_name} by {brand} is classified {tier} at {confidence:.0%} confidence"
                f"{er_note}, driven primarily by {semantic.get('tone')} tone "
                f"and {semantic.get('energy_level')} energy level."
            ),
        }


# ── Pydantic models ───────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    brand:              str
    media_name:         str
    duration:           float       = 0.0
    caption:            str         = ""
    img_summary:        str         = ""
    followers:          int         = 100_000
    is_collab:          bool        = False
    collaborators:      list[str]   = []
    created_at:         str         = ""
    top_k:              int         = 5
    similarity_threshold: float     = 0.50


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
def _startup() -> None:
    global _er_stats
    _load_llm_cache()
    _get_model()
    print("Model loaded.")
    _get_embed_model()
    print("Embedding model loaded.")
    _get_history()
    print(f"History embeddings ready ({_history_embeddings.shape[0]} posts).")
    # Compute ER reference stats directly from training history_df (already excludes val posts)
    global _er_media_stats
    if _history_df is not None:
        train_hist = _history_df[_history_df["tier"].isin(["LOW", "MEDIUM", "HIGH"])]
        # Tier-only aggregates
        for t in ["LOW", "MEDIUM", "HIGH"]:
            er_vals = train_hist[train_hist["tier"] == t]["engagement_rate"].dropna()
            if len(er_vals):
                _er_stats[t] = {
                    "median": round(float(er_vals.median()), 3),
                    "q25":    round(float(er_vals.quantile(0.25)), 3),
                    "q75":    round(float(er_vals.quantile(0.75)), 3),
                    "n":      len(er_vals),
                }
        # Tier × media_name breakdown
        for (t, mn), grp in train_hist.groupby(["tier", "media_name"]):
            er_vals = grp["engagement_rate"].dropna()
            if len(er_vals) >= 3:
                _er_media_stats[(t, mn)] = {
                    "median": round(float(er_vals.median()), 3),
                    "q25":    round(float(er_vals.quantile(0.25)), 3),
                    "q75":    round(float(er_vals.quantile(0.75)), 3),
                    "n":      len(er_vals),
                }
    print(f"ER stats (tier): {_er_stats}")
    print(f"ER stats (media×tier): {_er_media_stats}")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    _, feat_cols, _ = _get_model()
    return {
        "status":        "ok",
        "model":         MODEL_PATH.name,
        "features":      len(feat_cols),
        "history_posts": len(_history_df) if _history_df is not None else 0,
    }


@app.get("/validation_examples")
def validation_examples() -> list[dict]:
    """Return the 75 held-out validation posts from fold 4 with all raw fields."""
    if not FOLD_VAL_PATH.exists():
        raise HTTPException(status_code=404, detail="fold_4_val.parquet not found. Run train.ipynb.")
    if not DATASET_FINAL_PATH.exists():
        raise HTTPException(status_code=404, detail="dataset_final.parquet not found. Run train.ipynb.")

    # Load validation indices + actual tiers
    val_df   = pd.read_parquet(str(FOLD_VAL_PATH))   # index = row positions in dataset_final
    full_df  = pd.read_parquet(str(DATASET_FINAL_PATH))

    # Raw fields from assignment-dataset.json keyed by post_id
    with open(DATA_PATH) as f:
        raw = json.load(f)
    raw_map: dict[str, dict] = {}
    for item in raw:
        d  = item["data"]
        mc = d.get("metadata_content", {})
        ps = d.get("profile_stats", {})
        en = d.get("engagements", {})
        img_summary = ""
        for m in d.get("media", []):
            if m.get("type") == "thumbnail" and m.get("summary"):
                img_summary = m["summary"]
                break
        collaborators: list[str] = []
        for m in d.get("media", []):
            for c in m.get("collaborators", []) if isinstance(m.get("collaborators"), list) else []:
                if c:
                    collaborators.append(str(c))
        post_id = d.get("id", "")
        raw_map[post_id] = {
            "brand":          ps.get("username", ""),
            "media_name":     mc.get("media_name", ""),
            "caption":        mc.get("caption", ""),
            "img_summary":    img_summary,
            "followers":      int(ps.get("followers", 100_000)),
            "is_collab":      bool(d.get("is_collab") or mc.get("is_collab") or len(collaborators) > 0),
            "collaborators":  collaborators,
            "created_at":     mc.get("created_at", ""),
            "engagement_rate": float(en.get("engagement_rate", 0.0)),
            "likes":          int(en.get("likes", 0)),
            "views":          int(en.get("views", 0)),
        }

    examples = []
    for row_idx in val_df.index:
        row      = full_df.loc[row_idx]
        post_id  = str(row["post_id"])
        raw_info = raw_map.get(post_id, {})
        # Reconstruct duration from feature
        duration = float(row.get("duration", 0.0))
        # Reconstruct media_name from one-hot
        media_name = (
            "reel"  if row.get("media_name_reel",  0) else
            "album" if row.get("media_name_album", 0) else
            "post"
        )
        # Reconstruct brand from one-hot
        brand = next(
            (b.replace("brand_", "") for b in ["brand_cocacola_india","brand_pepsiindia",
             "brand_redbullindia","brand_sprite_india","brand_thumsupofficial"]
             if row.get(b, 0)), raw_info.get("brand", "")
        )
        examples.append({
            "post_id":        post_id,
            "brand":          brand,
            "media_name":     media_name,
            "duration":       duration,
            "caption":        raw_info.get("caption", ""),
            "img_summary":    raw_info.get("img_summary", ""),
            "followers":      raw_info.get("followers", 100_000),
            "is_collab":      raw_info.get("is_collab", False),
            "collaborators":  raw_info.get("collaborators", []),
            "created_at":     raw_info.get("created_at", ""),
            "actual_tier":    str(row["tier"]),
            "engagement_rate": raw_info.get("engagement_rate", 0.0),
            "likes":          raw_info.get("likes", 0),
            "views":          raw_info.get("views", 0),
        })

    return examples


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    model, feat_cols, explainer = _get_model()

    # 1. Structural feature engineering
    structural = _build_structural(
        brand          = req.brand,
        media_name     = req.media_name,
        duration       = req.duration,
        caption        = req.caption,
        followers      = req.followers,
        is_collab      = req.is_collab,
        collaborators  = req.collaborators,
        created_at_str = req.created_at or datetime.now(timezone.utc).isoformat(),
    )

    # 2. LLM semantic features
    semantic = _extract_semantic(req.caption, req.img_summary)

    # 3. Build aligned feature vector
    X = _build_feature_vector(structural, semantic, req.brand, req.media_name, feat_cols)

    # 4. XGBoost prediction
    tier_idx  = int(model.predict(X)[0])
    proba_arr = model.predict_proba(X)[0]
    tier      = TIER_LABELS[tier_idx]
    proba     = {TIER_LABELS[i]: round(float(p), 4) for i, p in enumerate(proba_arr)}

    # 5. SHAP — per-prediction feature importances
    try:
        shap_vals = explainer.shap_values(X)
        # Multi-class: list of (n_samples, n_features) per class
        if isinstance(shap_vals, list):
            sv = shap_vals[tier_idx][0]
        elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
            sv = shap_vals[0, :, tier_idx]
        else:
            sv = shap_vals[0]
        top_features = sorted(
            [{"feature": f, "shap_value": round(float(v), 4), "abs_shap": abs(float(v))}
             for f, v in zip(feat_cols, sv)],
            key=lambda x: x["abs_shap"],
            reverse=True,
        )[:10]
    except Exception:
        # Fallback to global feature importances
        imp = model.feature_importances_
        top_features = sorted(
            [{"feature": f, "shap_value": round(float(v), 4), "abs_shap": float(v)}
             for f, v in zip(feat_cols, imp)],
            key=lambda x: x["abs_shap"],
            reverse=True,
        )[:10]

    # 6. RAG — similar historical posts
    similar_posts = _retrieve_similar(
        caption    = req.caption,
        img_summary = req.img_summary,
        top_k       = req.top_k,
        threshold   = req.similarity_threshold,
    )

    # 7. LLM verdict + explanation
    llm_out = _generate_explanation(
        tier          = tier,
        proba         = proba,
        top_features  = top_features,
        similar_posts = similar_posts,
        semantic      = semantic,
        brand         = req.brand,
        media_name    = req.media_name,
        num_collabs   = len(req.collaborators),
    )

    return {
        "tier":          tier,
        "probabilities": proba,
        "verdict":       llm_out["verdict"],
        "explanation":   llm_out["explanation"],
        "er_reference":  {
            "tier_only":   _er_stats.get(tier, {}),
            "media_specific": _er_media_stats.get((tier, req.media_name.lower().strip()), {}),
            "media_name":  req.media_name,
        },
        "top_features":  top_features,
        "similar_posts": similar_posts,
        "semantic_features": {
            "content_theme":      semantic.get("content_theme", ""),
            "tone":               semantic.get("tone"),
            "language":           semantic.get("language"),
            "energy_level":       semantic.get("energy_level"),
            "production_quality": semantic.get("production_quality"),
            "product_prominence": semantic.get("product_prominence"),
            "cta_type":           semantic.get("cta_type"),
            "celebrity_presence": semantic.get("celebrity_presence"),
            "is_hinglish":        semantic.get("is_hinglish"),
            "has_question":       semantic.get("has_question"),
        },
    }
