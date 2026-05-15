import os
import streamlit as st

# Inject the Hugging Face token into the environment automatically
if "HF_TOKEN" in st.secrets:
    os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]

from hallucination_checker.scorer import check_answer

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Hallucination Checker",
    page_icon="🧠",
    layout="wide"
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .verdict-grounded {
        background: #d4edda; color: #155724;
        padding: 0.6rem 1.2rem; border-radius: 8px;
        font-size: 1.4rem; font-weight: 700;
        display: inline-block; margin-bottom: 1rem;
    }
    .verdict-partial {
        background: #fff3cd; color: #856404;
        padding: 0.6rem 1.2rem; border-radius: 8px;
        font-size: 1.4rem; font-weight: 700;
        display: inline-block; margin-bottom: 1rem;
    }
    .verdict-hallucinated {
        background: #f8d7da; color: #721c24;
        padding: 0.6rem 1.2rem; border-radius: 8px;
        font-size: 1.4rem; font-weight: 700;
        display: inline-block; margin-bottom: 1rem;
    }
    .claim-card {
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.8rem;
        background: #f8f9fa;
    }
    .label-entailment { color: #155724; font-weight: 600; }
    .label-neutral    { color: #856404; font-weight: 600; }
    .label-contradiction { color: #721c24; font-weight: 600; }
    .evidence-text {
        font-size: 0.85rem; color: #495057;
        border-left: 3px solid #adb5bd;
        padding-left: 0.8rem; margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🧠 LLM Hallucination Checker")
st.markdown(
    "Paste any AI-generated answer below. This tool splits it into individual "
    "factual claims, retrieves Wikipedia evidence, and uses an NLI model to "
    "verify each claim — producing an interpretable hallucination verdict."
)
st.divider()

# ── Input ─────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 1])

with col1:
    answer = st.text_area(
        "AI-Generated Answer",
        placeholder="Paste the LLM output you want to fact-check here...",
        height=180
    )

with col2:
    subject = st.text_input(
        "Topic / Subject",
        placeholder="e.g. Alan Turing, Black holes"
    )
    st.markdown("&nbsp;", unsafe_allow_html=True)
    run = st.button("🔍 Check for Hallucinations", use_container_width=True, type="primary")

# ── Example ───────────────────────────────────────────────────────────────────
with st.expander("💡 Try an example"):
    st.markdown("**Subject:** Alan Turing")
    st.markdown(
        "**Answer:** Alan Turing was a British mathematician and computer scientist. "
        "He invented the iPhone in 1950. He also worked at Bletchley Park during World War II "
        "and developed the Turing Test to evaluate machine intelligence."
    )
    if st.button("Load this example"):
        st.session_state["example_loaded"] = True
        st.rerun()

if st.session_state.get("example_loaded"):
    answer = (
        "Alan Turing was a British mathematician and computer scientist. "
        "He invented the iPhone in 1950. He also worked at Bletchley Park during World War II "
        "and developed the Turing Test to evaluate machine intelligence."
    )
    subject = "Alan Turing"
    run = True
    st.session_state["example_loaded"] = False

# ── Run pipeline ──────────────────────────────────────────────────────────────
if run:
    if not answer.strip():
        st.warning("Please paste an AI-generated answer first.")
    elif not subject.strip():
        st.warning("Please enter the topic or subject of the answer.")
    else:
        with st.spinner("Fetching Wikipedia evidence and running NLI model... (this takes ~20-30s on first run)"):
            try:
                verdict, score, claim_results = check_answer(answer.strip(), subject.strip())
                if verdict == "Unknown":
                    st.error(f"Could not fetch Wikipedia article for: '{subject}'. Try a different spelling.")
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.stop()

        st.divider()

        # ── Verdict banner ────────────────────────────────────────────────────
        if verdict == "Grounded":
            css_class = "verdict-grounded"
            emoji = "✅"
        elif verdict == "Partially Grounded":
            css_class = "verdict-partial"
            emoji = "⚠️"
        elif verdict == "Hallucinated":
            css_class = "verdict-hallucinated"
            emoji = "🚨"
        else:
            css_class = "verdict-partial"
            emoji = "❓"

        st.markdown(
            f'<div class="{css_class}">{emoji} Verdict: {verdict}</div>',
            unsafe_allow_html=True
        )

        # ── Score bar ─────────────────────────────────────────────────────────
        m1, m2, m3 = st.columns(3)
        m1.metric("Hallucination Score", f"{score:.2f}", help="0 = fully grounded, 1 = fully hallucinated")
        m2.metric("Claims Checked", len(claim_results))
        contradictions = sum(1 for r in claim_results if r["label"] == "CONTRADICTION")
        m3.metric("Contradictions Found", contradictions)

        st.progress(score)

        # ── Claim-by-claim breakdown ──────────────────────────────────────────
        st.markdown("### 📋 Claim-by-Claim Breakdown")

        label_map = {
            "ENTAILMENT":    ("✅ Entailment",    "label-entailment"),
            "NEUTRAL":       ("⚠️ Neutral",        "label-neutral"),
            "CONTRADICTION": ("🚨 Contradiction",  "label-contradiction"),
        }

        for i, result in enumerate(claim_results, 1):
            label_display, label_css = label_map.get(
                result["label"], (result["label"], "")
            )
            with st.container():
                st.markdown(f"""
                <div class="claim-card">
                    <strong>Claim {i}:</strong> {result['claim']}<br>
                    <span class="{label_css}">{label_display}</span>
                    &nbsp;&nbsp;
                    <span style="color:#6c757d; font-size:0.85rem;">
                        Confidence: {result['confidence']:.0%}
                    </span>
                    <div class="evidence-text">
                        <strong>Evidence used:</strong> {result['evidence'][:300]}{'...' if len(result['evidence']) > 300 else ''}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # ── Footer note ───────────────────────────────────────────────────────
        st.divider()
        st.caption(
            "Evidence sourced from Wikipedia. NLI model: facebook/bart-large-mnli. "
            "Embeddings: all-MiniLM-L6-v2. This tool is for research and evaluation purposes."
        )