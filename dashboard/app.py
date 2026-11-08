"""
ReviewMatrix - Streamlit Dashboard
--------------------------------------
Interactive dashboard: pick a product, see its aspect-wise sentiment
breakdown as a chart, read example negative reviews, and view an
AI-generated executive summary.

IMPORTANT: This app never imports psycopg2 or langchain_nvidia_ai_endpoints
directly. All database and LLM work happens in separate subprocesses
(pipeline/aggregate_runner.py and pipeline/summarizer.py) to avoid a native
DLL conflict between the two libraries on Windows.
"""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PIPELINE_DIR = Path(__file__).parent.parent / "pipeline"

st.set_page_config(page_title="ReviewMatrix", layout="wide")


def run_subprocess(*args) -> str:
    """Runs aggregate_runner.py as a subprocess inside the pipeline folder."""
    result = subprocess.run(
        [sys.executable, "aggregate_runner.py", *args],
        capture_output=True,
        text=True,
        cwd=str(PIPELINE_DIR),
    )
    if result.returncode != 0:
        raise RuntimeError(f"aggregate_runner.py failed:\n{result.stderr}")
    return result.stdout.strip()


def run_summarizer(product_id: str) -> str:
    """Runs summarizer.py as a subprocess to get an LLM-generated summary."""
    result = subprocess.run(
        [sys.executable, "summarizer.py", product_id],
        capture_output=True,
        text=True,
        cwd=str(PIPELINE_DIR),
    )
    if result.returncode != 0:
        raise RuntimeError(f"summarizer.py failed:\n{result.stderr}")
    return result.stdout.strip()


@st.cache_data(ttl=300)
def get_product_list(limit: int = 30):
    raw = run_subprocess("--list", str(limit))
    return json.loads(raw)


@st.cache_data(ttl=60)
def get_counts(product_id: str):
    raw = run_subprocess("--counts", product_id)
    return json.loads(raw)


@st.cache_data(ttl=60)
def get_negatives(product_id: str, aspect: str):
    raw = run_subprocess("--negatives", product_id, aspect)
    return json.loads(raw)


@st.cache_data(ttl=300)
def get_summary(product_id: str):
    return run_summarizer(product_id)


# ---------------- UI ----------------

st.title("ReviewMatrix")
st.caption("Aspect-Based Sentiment Analysis for Product Reviews")

with st.spinner("Loading product list..."):
    try:
        products = get_product_list(30)
    except Exception as e:
        st.error(f"Could not load product list. Is PostgreSQL running?\n\n{e}")
        st.stop()

if not products:
    st.warning("No products found in the database. Run the pipeline scripts first.")
    st.stop()

product_options = {
    f"{p['product_id']}  ({p['review_count']} reviews)": p["product_id"]
    for p in products
}

selected_label = st.selectbox("Select a product", list(product_options.keys()))
selected_product = product_options[selected_label]

st.divider()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Aspect-wise Sentiment Breakdown")
    with st.spinner("Loading sentiment data..."):
        try:
            counts_data = get_counts(selected_product)
        except Exception as e:
            st.error(f"Failed to load sentiment data:\n\n{e}")
            st.stop()

    if not counts_data:
        st.info("No aspect-sentiment data found for this product.")
    else:
        df = pd.DataFrame(counts_data)
        pivot = df.pivot_table(
            index="aspect", columns="sentiment", values="count", fill_value=0
        )
        # Ensure consistent column order/colors if present
        for col in ["Positive", "Neutral", "Negative"]:
            if col not in pivot.columns:
                pivot[col] = 0
        pivot = pivot[["Positive", "Neutral", "Negative"]]
        st.bar_chart(pivot, color=["#2ecc71", "#95a5a6", "#e74c3c"])

with col2:
    st.subheader("AI Summary")
    with st.spinner("Generating summary with Llama 3.1..."):
        try:
            summary = get_summary(selected_product)
            st.write(summary)
        except Exception as e:
            st.error(f"Failed to generate summary:\n\n{e}")

st.divider()

st.subheader("Example Negative Reviews")
if not counts_data:
    st.info("No data available.")
else:
    aspects_available = sorted({row["aspect"] for row in counts_data})
    selected_aspect = st.selectbox("Select an aspect to inspect", aspects_available)

    with st.spinner("Loading example reviews..."):
        try:
            negatives = get_negatives(selected_product, selected_aspect)
        except Exception as e:
            st.error(f"Failed to load examples:\n\n{e}")
            negatives = []

    if not negatives:
        st.success(f"No negative mentions found for '{selected_aspect}'.")
    else:
        for row in negatives:
            st.markdown(f"- *\"{row['sentence']}\"* (confidence: {row['sentiment_confidence']:.2f})")
