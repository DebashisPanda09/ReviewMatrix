"""
ReviewMatrix - Aggregation Runner (subprocess-safe)
-------------------------------------------------------
Only imports psycopg2/pandas — never langchain_nvidia_ai_endpoints.
Meant to be invoked as a subprocess so the database query (psycopg2) and
the LLM call (langchain-nvidia, which pulls in aiohttp) never coexist in
the same process. This avoids a native DLL/OpenSSL conflict between the
two libraries on Windows.

Usage:
    python aggregate_runner.py --top
        -> prints the top product's ID (plain text)

    python aggregate_runner.py --list [N]
        -> prints JSON list of top N products with review counts

    python aggregate_runner.py --counts <product_id>
        -> prints JSON list of {aspect, sentiment, count} for charting

    python aggregate_runner.py --negatives <product_id> <aspect>
        -> prints JSON list of example negative sentences

    python aggregate_runner.py <product_id>
        -> prints formatted stats text (for the LLM prompt)
"""

import sys
import json
from aggregate import (
    format_stats_for_summary,
    get_top_products,
    get_aspect_sentiment_counts,
    get_negative_examples,
)


def main():
    if len(sys.argv) < 2:
        print("ERROR: missing argument", file=sys.stderr)
        sys.exit(1)

    command = sys.argv[1]

    if command == "--top":
        top = get_top_products(limit=1)
        if top.empty:
            print("ERROR: no products found", file=sys.stderr)
            sys.exit(1)
        print(top.iloc[0]["product_id"])
        return

    if command == "--list":
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        top = get_top_products(limit=limit)
        print(json.dumps(top.to_dict(orient="records")))
        return

    if command == "--counts":
        if len(sys.argv) < 3:
            print("ERROR: missing product_id", file=sys.stderr)
            sys.exit(1)
        product_id = sys.argv[2]
        df = get_aspect_sentiment_counts(product_id)
        print(json.dumps(df.to_dict(orient="records")))
        return

    if command == "--negatives":
        if len(sys.argv) < 4:
            print("ERROR: missing product_id or aspect", file=sys.stderr)
            sys.exit(1)
        product_id = sys.argv[2]
        aspect = sys.argv[3]
        df = get_negative_examples(product_id, aspect, limit=5)
        print(json.dumps(df.to_dict(orient="records")))
        return

    # Default: treat the argument as a product_id, return formatted stats text
    product_id = command
    stats = format_stats_for_summary(product_id)
    print(stats)


if __name__ == "__main__":
    main()
