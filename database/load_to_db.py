"""
ReviewMatrix - Load to PostgreSQL
-------------------------------------
Loads aspect_sentiments.csv into the PostgreSQL `reviewmatrix` database,
populating both the `reviews` and `aspect_sentiments` tables.

Requires: database/schema.sql to have been run first (creates the tables).
Reads DB credentials from .env
"""

import os
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

INPUT_PATH = "data/processed/aspect_sentiments.csv"

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
}


def main():
    print(f"Loading data from {INPUT_PATH} ...")
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} rows")

    print("Connecting to PostgreSQL...")
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # ---- Insert unique reviews first (parent table) ----
    print("Preparing unique reviews...")
    reviews_df = df[["review_id", "product_id", "rating"]].drop_duplicates(
        subset=["review_id"]
    )
    review_rows = list(
        reviews_df.itertuples(index=False, name=None)
    )  # (review_id, product_id, rating)

    print(f"Inserting {len(review_rows)} unique reviews...")
    execute_values(
        cur,
        """
        INSERT INTO reviews (review_id, product_id, rating)
        VALUES %s
        ON CONFLICT (review_id) DO NOTHING
        """,
        review_rows,
        page_size=1000,
    )
    conn.commit()
    print("Reviews inserted.")

    # ---- Insert aspect_sentiments (child table) ----
    print("Preparing aspect-sentiment rows...")
    aspect_rows = list(
        df[
            [
                "review_id",
                "product_id",
                "sentence",
                "aspect",
                "sentiment",
                "sentiment_confidence",
                "aspect_confidence",
            ]
        ].itertuples(index=False, name=None)
    )

    print(f"Inserting {len(aspect_rows)} aspect-sentiment rows...")
    batch_size = 2000
    for start in tqdm(range(0, len(aspect_rows), batch_size)):
        batch = aspect_rows[start : start + batch_size]
        execute_values(
            cur,
            """
            INSERT INTO aspect_sentiments
                (review_id, product_id, sentence, aspect, sentiment,
                 sentiment_confidence, aspect_confidence)
            VALUES %s
            """,
            batch,
            page_size=1000,
        )
        conn.commit()

    print("All aspect-sentiment rows inserted.")

    # ---- Quick verification ----
    cur.execute("SELECT COUNT(*) FROM reviews;")
    print(f"Total rows in reviews table: {cur.fetchone()[0]}")

    cur.execute("SELECT COUNT(*) FROM aspect_sentiments;")
    print(f"Total rows in aspect_sentiments table: {cur.fetchone()[0]}")

    cur.close()
    conn.close()
    print("Done. Connection closed.")


if __name__ == "__main__":
    main()
