"""
frontend.py — Social Media Performance Predictor
Professional Streamlit UI. Two modes: Validation Testing | New Post Prediction.
"""
from __future__ import annotations
from datetime import datetime, time as dt_time
import pandas as pd
import requests
import streamlit as st

API_URL      = "http://localhost:8000"
KNOWN_BRANDS = ["sprite_india", "cocacola_india", "pepsiindia", "redbullindia", "thumsupofficial"]
TIER_COLOR   = {"LOW": "#e74c3c", "MEDIUM": "#e67e22", "HIGH": "#2ecc71"}
TIER_EMOJI   = {"LOW": "🔴", "MEDIUM": "🟡", "HIGH": "🟢"}

# ── Page ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Content Performance Predictor", page_icon="📊",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
/* ── global ── */
[data-testid="stAppViewContainer"] { background: #0f1117; }
[data-testid="stHeader"] { background: transparent; }

/* ── verdict card ── */
.verdict-card {
    border-radius: 12px;
    padding: 1.1rem 1.4rem;
    margin: 0.8rem 0 1.2rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.verdict-label {
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    opacity: 0.7;
    margin-bottom: 0.15rem;
}
.verdict-text {
    font-size: 1.3rem;
    font-weight: 700;
    line-height: 1.2;
}
.tier-pill {
    display: inline-block;
    padding: 0.25em 0.85em;
    border-radius: 999px;
    font-size: 1.05rem;
    font-weight: 800;
    letter-spacing: 0.05em;
    border: 2px solid currentColor;
}
/* ── prob bar ── */
.prob-row { display: flex; align-items: center; gap: 0.6rem; margin: 0.2rem 0; font-size: 0.88rem; }
.prob-label { width: 5.5rem; opacity: 0.8; }
.prob-bar-wrap { flex: 1; background: rgba(255,255,255,0.08); border-radius: 4px; height: 8px; }
.prob-bar { height: 8px; border-radius: 4px; }
.prob-val { width: 3.2rem; text-align: right; font-weight: 600; }
/* ── match badge ── */
.match-ok  { color: #2ecc71; font-weight: 700; font-size: 1.05rem; }
.match-bad { color: #e74c3c; font-weight: 700; font-size: 1.05rem; }
/* ── feature row ── */
.feat-row { display: flex; align-items: center; gap: 0.5rem; margin: 0.18rem 0; font-size: 0.82rem; }
.feat-name { width: 14rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; opacity: 0.85; }
.feat-bar-wrap { flex: 1; background: rgba(255,255,255,0.06); border-radius: 3px; height: 6px; position: relative; }
.feat-zero { position: absolute; left: 50%; top: 0; width: 1px; height: 6px; background: rgba(255,255,255,0.25); }
.feat-bar { height: 6px; border-radius: 3px; position: absolute; top: 0; }
.feat-val { width: 3.8rem; text-align: right; font-weight: 600; font-family: monospace; }
/* ── attr chip ── */
.chip { display: inline-block; padding: 0.2em 0.65em; border-radius: 6px;
        background: rgba(255,255,255,0.08); font-size: 0.78rem; margin: 0.18rem 0.18rem 0 0; }
.chip-on { background: rgba(46,204,113,0.18); color: #2ecc71; }
/* ── similar post ── */
.sim-card { border-left: 3px solid rgba(255,255,255,0.15); padding: 0.5rem 0.8rem;
            margin: 0.4rem 0; border-radius: 0 6px 6px 0; background: rgba(255,255,255,0.03); }
.sim-header { font-size: 0.8rem; opacity: 0.6; margin-bottom: 0.25rem; }
.sim-caption { font-size: 0.85rem; line-height: 1.4; }
/* ── divider ── */
.section-title { font-size: 0.7rem; letter-spacing: 0.12em; text-transform: uppercase;
                 opacity: 0.5; margin: 1.4rem 0 0.5rem; }
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
hcol, scol = st.columns([7, 1])
with hcol:
    st.markdown("## 📊 Content Performance Predictor")
    st.caption("XGBoost · 5-Fold CV · Macro-F1 0.424 · SHAP · RAG · LLM verdict")
with scol:
    try:
        h = requests.get(f"{API_URL}/health", timeout=2).json()
        n = h.get("history_posts", "?")
        st.markdown(f"<div style='text-align:right;padding-top:1rem'>"
                    f"<span style='color:#2ecc71'>● online</span> "
                    f"<span style='opacity:.5;font-size:.8rem'>{n} posts indexed</span></div>",
                    unsafe_allow_html=True)
    except Exception:
        st.markdown("<div style='text-align:right;padding-top:1rem'>"
                    "<span style='color:#e74c3c'>● offline</span></div>",
                    unsafe_allow_html=True)

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_val, tab_new = st.tabs(["🧪 Validation Set", "🔮 New Post"])


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED results renderer
# ═══════════════════════════════════════════════════════════════════════════════
def _render_results(result: dict, actual_tier: str | None = None) -> None:
    tier    = result["tier"]
    proba   = result["probabilities"]
    verdict = result.get("verdict", tier)
    expl    = result.get("explanation", "")
    sem     = result.get("semantic_features", {})
    feats   = result.get("top_features", [])
    posts   = result.get("similar_posts", [])
    er_ref  = result.get("er_reference", {})
    # Prefer media-specific stats, fall back to tier-only
    _er_ms   = er_ref.get("media_specific", {})
    _er_ts   = er_ref.get("tier_only", {})
    _er_data = _er_ms if _er_ms else _er_ts
    _er_mn   = er_ref.get("media_name", "")
    _er_label = f"Typical ER ({tier} {_er_mn})" if _er_ms else f"Typical ER ({tier})"
    color   = TIER_COLOR[tier]

    # ── Verdict card ──────────────────────────────────────────────────────────
    conf = proba.get(tier, 0)
    match_html = ""
    if actual_tier:
        ok = tier == actual_tier
        cls = "match-ok" if ok else "match-bad"
        txt = ("✅ Correct" if ok else f"❌ Wrong — actual {TIER_EMOJI[actual_tier]} {actual_tier}")
        match_html = f'<span class="{cls}" style="margin-left:1rem">{txt}</span>'

    # ER reference block for verdict card
    er_html = ""
    if _er_data:
        small_n = _er_data.get("n", 99) < 15
        caveat  = ' <span style="font-size:.65rem;opacity:.5">⚠ small sample</span>' if small_n else ""
        er_html = (
            f'<div style="text-align:right;white-space:nowrap;margin-left:1.5rem">'
            f'<div class="verdict-label">{_er_label}</div>'
            f'<div style="font-size:1.05rem;font-weight:700;color:{color}">'
            f'{_er_data["median"]:.2f}%{caveat}</div>'
            f'<div style="font-size:.72rem;opacity:.6">'
            f'IQR {_er_data["q25"]:.2f}%\u2013{_er_data["q75"]:.2f}%'
            f' &nbsp;n={_er_data["n"]}</div>'
            f'</div>'
        )

    st.markdown(
        f'<div class="verdict-card" style="background:rgba({_hex_to_rgb(color)},0.12);'
        f'border:1px solid rgba({_hex_to_rgb(color)},0.35)">'
        f'<div style="flex:0 0 auto">'
        f'<span class="tier-pill" style="color:{color}">{TIER_EMOJI[tier]} {tier}</span>'
        f'</div>'
        f'<div style="flex:1;min-width:0">'
        f'<div class="verdict-label">Analyst Verdict</div>'
        f'<div class="verdict-text">{verdict}</div>'
        f'</div>'
        f'<div style="text-align:right;white-space:nowrap">'
        f'<div class="verdict-label">Confidence</div>'
        f'<div style="font-size:1.6rem;font-weight:800;color:{color}">{conf:.0%}</div>'
        f'</div>'
        f'{er_html}'
        f'</div>'
        f'{match_html}',
        unsafe_allow_html=True,
    )

    # ── Two-column layout ─────────────────────────────────────────────────────
    left, right = st.columns([5, 4], gap="large")

    with left:
        # Probability bars
        st.markdown('<div class="section-title">Probability Distribution</div>',
                    unsafe_allow_html=True)
        for lbl, p in proba.items():
            c = TIER_COLOR[lbl]
            bar_pct = int(p * 100)
            st.markdown(
                f'<div class="prob-row">'
                f'<span class="prob-label">{TIER_EMOJI[lbl]} {lbl}</span>'
                f'<div class="prob-bar-wrap">'
                f'<div class="prob-bar" style="width:{bar_pct}%;background:{c}"></div>'
                f'</div>'
                f'<span class="prob-val" style="color:{c}">{p:.1%}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # SHAP importances
        st.markdown('<div class="section-title">Key Predictive Signals (SHAP)</div>',
                    unsafe_allow_html=True)
        if feats:
            max_abs = max(abs(f["shap_value"]) for f in feats) or 1.0
            for f in feats[:8]:
                v   = f["shap_value"]
                pct = abs(v) / max_abs * 50          # half-bar = 50% of container
                pos = v > 0
                bar_color = "#2ecc71" if pos else "#e74c3c"
                # positive bars grow right from center, negative grow left
                bar_left  = 50 if pos else (50 - pct)
                st.markdown(
                    f'<div class="feat-row">'
                    f'<span class="feat-name" title="{f["feature"]}">{f["feature"]}</span>'
                    f'<div class="feat-bar-wrap">'
                    f'<div class="feat-zero"></div>'
                    f'<div class="feat-bar" style="width:{pct}%;left:{bar_left}%;background:{bar_color}"></div>'
                    f'</div>'
                    f'<span class="feat-val" style="color:{bar_color}">{v:+.3f}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # Explanation
        st.markdown('<div class="section-title">Classification Rationale</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:1.05rem;line-height:1.8;padding:1rem 1.1rem;'
            f'background:rgba(255,255,255,0.04);border-radius:10px;'
            f'border-left:3px solid {color};margin-top:.3rem">{expl}</div>',
            unsafe_allow_html=True,
        )

    with right:
        # Content attributes
        st.markdown('<div class="section-title">Detected Content Attributes</div>',
                    unsafe_allow_html=True)
        bool_fields = {"celebrity_presence": "Celebrity", "is_hinglish": "Hinglish",
                       "has_question": "Question CTA"}
        str_fields  = {"tone": None, "language": None, "energy_level": "Energy",
                       "production_quality": "Production", "cta_type": "CTA",
                       "content_theme": "Theme"}
        chips = []
        for k, label in str_fields.items():
            v = sem.get(k)
            if v:
                chips.append(f'<span class="chip">{label or k.replace("_"," ").title()}: <b>{v}</b></span>')
        for k, label in bool_fields.items():
            if sem.get(k):
                chips.append(f'<span class="chip chip-on">✓ {label}</span>')
        st.markdown("".join(chips) or "<span style='opacity:.4'>—</span>",
                    unsafe_allow_html=True)

        # Similar posts
        st.markdown(
            f'<div class="section-title">Similar Historical Posts '
            f'<span style="opacity:.5">({len(posts)} matched)</span></div>',
            unsafe_allow_html=True,
        )
        if posts:
            for p in posts:
                p_tier  = p.get("tier", "")
                t_color = TIER_COLOR.get(p_tier, "#aaa")
                t_emoji = TIER_EMOJI.get(p_tier, "")
                sim_color = "#2ecc71" if p["similarity"] >= 0.7 else \
                            "#e67e22" if p["similarity"] >= 0.5 else "#aaa"
                tier_pill = (
                    f'<span style="color:{t_color};border:1px solid {t_color};'
                    f'border-radius:4px;padding:.1em .45em;font-size:.72rem;'
                    f'font-weight:700;margin-right:.4rem">{t_emoji} {p_tier}</span>'
                    if p_tier else ""
                )
                st.markdown(
                    f'<div class="sim-card">'  
                    f'<div class="sim-header">'  
                    f'{tier_pill}<b>{p["brand"]}</b> · {p["media_name"]} · '  
                    f'ER <b>{p["engagement_rate"]:.2f}</b> · '
                    f'<span style="color:{sim_color}">sim {p["similarity"]:.2f}</span>'
                    f'</div>'
                    f'<div class="sim-caption">{p["caption_snippet"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown("<span style='opacity:.4;font-size:.85rem'>No posts above threshold.</span>",
                        unsafe_allow_html=True)


def _hex_to_rgb(h: str) -> str:
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{r},{g},{b}"


def _call_predict(payload: dict) -> dict:
    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=90)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("Backend offline — run `python main.py`")
        st.stop()
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Validation Set
# ═══════════════════════════════════════════════════════════════════════════════
with tab_val:
    st.markdown(
        "<span style='opacity:.6;font-size:.85rem'>"
        "75 held-out posts from fold 4 (never seen during training). "
        "Select one to run the full pipeline and compare predicted vs actual tier."
        "</span>",
        unsafe_allow_html=True,
    )
    st.write("")

    @st.cache_data(ttl=600, show_spinner="Loading validation set…")
    def _load_val() -> list[dict]:
        r = requests.get(f"{API_URL}/validation_examples", timeout=30)
        r.raise_for_status()
        return r.json()

    try:
        val_examples = _load_val()
    except Exception as e:
        st.error(f"Could not load validation set: {e}")
        st.stop()

    # Filters
    fc1, fc2, fc3 = st.columns(3)
    f_brand = fc1.selectbox("Brand", ["All"] + KNOWN_BRANDS, key="vb")
    f_media = fc2.selectbox("Media", ["All", "reel", "post", "album"], key="vm")
    f_tier  = fc3.selectbox("Actual Tier", ["All", "HIGH", "MEDIUM", "LOW"], key="vt")

    shown = [
        ex for ex in val_examples
        if (f_brand == "All" or ex["brand"] == f_brand)
        and (f_media == "All" or ex["media_name"] == f_media)
        and (f_tier  == "All" or ex["actual_tier"] == f_tier)
    ]
    st.caption(f"{len(shown)} of {len(val_examples)} examples")

    if not shown:
        st.warning("No examples match these filters.")
        st.stop()

    # Compact table
    tdf = pd.DataFrame([
        {"#": i+1, "Brand": e["brand"], "Type": e["media_name"],
         "Actual": e["actual_tier"], "ER": round(e["engagement_rate"], 3),
         "Caption": (e.get("caption") or "")[:60] + ("…" if len(e.get("caption",""))>60 else "")}
        for i, e in enumerate(shown)
    ])

    def _ct(v: str) -> str:
        return {"HIGH": "color:#2ecc71;font-weight:700",
                "MEDIUM": "color:#e67e22;font-weight:700",
                "LOW": "color:#e74c3c;font-weight:700"}.get(v, "")

    st.dataframe(
        tdf.style.map(_ct, subset=["Actual"]),
        hide_index=True, use_container_width=True,
        height=min(36 * len(shown) + 38, 300),
    )

    # Selector + preview
    sel_i = st.selectbox(
        "Select example",
        range(1, len(shown)+1),
        format_func=lambda i: (
            f"#{i}  {shown[i-1]['brand']} · {shown[i-1]['media_name']} "
            f"· {shown[i-1]['actual_tier']} · ER {shown[i-1]['engagement_rate']:.3f}"
        ),
        key="vs",
    )
    sel = shown[sel_i - 1]

    with st.expander("Post details", expanded=False):
        pa, pb, pc, pd_ = st.columns(4)
        pa.metric("Brand",  sel["brand"])
        pb.metric("Type",   sel["media_name"])
        pc.metric("Actual", sel["actual_tier"])
        pd_.metric("ER",    f"{sel['engagement_rate']:.3f}")
        if sel.get("caption"):
            st.text_area("Caption",       sel["caption"],     height=90,  disabled=True, key="vc")
        if sel.get("img_summary"):
            st.text_area("Image summary", sel["img_summary"], height=70,  disabled=True, key="vi")

    ra, rb = st.columns(2)
    v_k = ra.slider("Top-K similar posts", 1, 10, 5, key="vk")
    v_t = rb.slider("Similarity threshold", 0.0, 1.0, 0.65, 0.05, key="vth")

    if st.button("Analyze", type="primary", use_container_width=True, key="vrun"):
        with st.spinner("Running pipeline…"):
            result = _call_predict({
                "brand": sel["brand"], "media_name": sel["media_name"],
                "duration": sel["duration"], "caption": sel["caption"],
                "img_summary": sel["img_summary"], "followers": sel["followers"],
                "is_collab": sel["is_collab"], "collaborators": sel["collaborators"],
                "created_at": sel["created_at"],
                "top_k": v_k, "similarity_threshold": v_t,
            })
        st.divider()
        _render_results(result, actual_tier=sel["actual_tier"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — New Post
# ═══════════════════════════════════════════════════════════════════════════════
with tab_new:
    with st.form("nf"):
        r1a, r1b, r1c = st.columns([2, 2, 2])
        brand      = r1a.selectbox("Brand", KNOWN_BRANDS)
        media_name = r1b.selectbox("Type", ["reel", "post", "album"])
        followers  = r1c.number_input("Followers", 1_000, 10_000_000, 170_000, 5_000)

        r2a, r2b, r2c = st.columns([2, 1, 1])
        duration  = r2a.slider("Reel duration (s)", 5, 180, 30, 5,
                               help="Ignored unless Type = reel")
        post_date = r2b.date_input("Date", datetime.now().date())
        post_time = r2c.time_input("Time", dt_time(12, 0))
        created_at = datetime.combine(post_date, post_time).isoformat()

        is_collab   = st.checkbox("Collaboration post")
        collabs_raw = st.text_input("Collaborator handles (comma-separated)",
                                    placeholder="@handle1, @handle2",
                                    disabled=not is_collab)
        collaborators = [c.strip().lstrip("@") for c in collabs_raw.split(",") if c.strip()]

        t1, t2 = st.columns(2)
        caption     = t1.text_area("Caption", height=140,
                                    placeholder="Paste the Instagram caption…")
        img_summary = t2.text_area("Image / video description", height=140,
                                    placeholder="Describe the visual: scene, product placement, text overlays…")

        with st.expander("Retrieval settings"):
            ra2, rb2 = st.columns(2)
            top_k      = ra2.slider("Top-K similar posts", 1, 10, 5)
            sim_thresh = rb2.slider("Similarity threshold", 0.0, 1.0, 0.50, 0.05)

        go = st.form_submit_button("Analyze Post", type="primary", use_container_width=True)

    if go:
        if not caption.strip() and not img_summary.strip():
            st.warning("Provide at least a caption or image description.")
            st.stop()
        with st.spinner("Running pipeline…"):
            result = _call_predict({
                "brand": brand, "media_name": media_name,
                "duration": float(duration) if media_name == "reel" else 0.0,
                "caption": caption, "img_summary": img_summary,
                "followers": int(followers), "is_collab": is_collab,
                "collaborators": collaborators, "created_at": created_at,
                "top_k": top_k, "similarity_threshold": sim_thresh,
            })
        st.divider()
        _render_results(result)
