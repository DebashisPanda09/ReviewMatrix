"""
ReviewMatrix - Aggregation Module
-------------------------------------
Provides functions to pull aspect-wise sentiment breakdowns from PostgreSQL,
either for a single product or across the whole dataset.

Used by:
- pipeline/summarizer.py (feeds formatted stats into the LLM)
- dashboard/app.py (feeds chart data)
"""

import os
import psycopg2
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
}


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def get_top_products(limit: int = 20) -> pd.DataFrame:
    """Returns the products with the most review mentions, for dashboard dropdown."""
    query = """
        SELECT product_id, COUNT(DISTINCT review_id) AS review_count
        FROM aspect_sentiments
        GROUP BY product_id
        ORDER BY review_count DESC
        LIMIT %s;
    """
    conn = get_connection()
    df = pd.read_sql(query, conn, params=(limit,))
    conn.close()
    return df


def get_aspect_sentiment_counts(product_id: str) -> pd.DataFrame:
    """Returns raw aspect x sentiment counts for a given product_id."""
    query = """
        SELECT aspect, sentiment, COUNT(*) AS count
        FROM aspect_sentiments
        WHERE product_id = %s
        GROUP BY aspect, sentiment
        ORDER BY aspect, sentiment;
    """
    conn = get_connection()
    df = pd.read_sql(query, conn, params=(product_id,))
    conn.close()
    return df


def get_negative_examples(product_id: str, aspect: str, limit: int = 5) -> pd.DataFrame:
    """Returns example negative sentences for a given product + aspect (for dashboard drill-down)."""
    query = """
        SELECT sentence, sentiment_confidence
        FROM aspect_sentiments
        WHERE product_id = %s AND aspect = %s AND sentiment = 'Negative'
        ORDER BY sentiment_confidence DESC
        LIMIT %s;
    """
    conn = get_connection()
    df = pd.read_sql(query, conn, params=(product_id, aspect, limit))
    conn.close()
    return df


def format_stats_for_summary(product_id: str) -> str:
    """
    Builds a compact, human-readable string of aspect-sentiment percentages
    for a given product, ready to feed into the LLM summarizer prompt.
    """
    df = get_aspect_sentiment_counts(product_id)
    if df.empty:
        return f"No review data found for product {product_id}."

    lines = [f"Aspect sentiment breakdown for product {product_id}:"]
    for aspect, group in df.groupby("aspect"):
        total = group["count"].sum()
        parts = []
        for _, row in group.iterrows():
            pct = round((row["count"] / total) * 100)
            parts.append(f"{row['sentiment']} {pct}%")
        lines.append(f"- {aspect} ({total} mentions): " + ", ".join(parts))

    return "\n".join(lines)


if __name__ == "__main__":
    # Quick manual test: show stats for the product with the most reviews
    top = get_top_products(limit=1)
    if not top.empty:
        pid = top.iloc[0]["product_id"]
        print(f"Testing aggregation for top product: {pid}\n")
        print(format_stats_for_summary(pid))
    else:
        print("No products found in database.")
