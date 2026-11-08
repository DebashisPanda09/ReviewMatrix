"""
ReviewMatrix - Preprocessing Script
------------------------------------
Loads the raw Amazon Fine Food Reviews dataset, samples a manageable subset,
cleans the text, and splits each review into sentences using SpaCy.

Input:  data/raw/Reviews.csv
Output: data/processed/cleaned_reviews.csv
        Columns: review_id, product_id, sentence, original_review, rating
"""

import re
import pandas as pd
import spacy
from tqdm import tqdm

# ---- Config ----
INPUT_PATH = "data/raw/Reviews.csv"
OUTPUT_PATH = "data/processed/cleaned_reviews.csv"
SAMPLE_SIZE = 5000       # number of reviews to sample (adjust if needed)
RANDOM_STATE = 42        # for reproducibility

# ---- Load SpaCy model ----
print("Loading SpaCy model...")
nlp = spacy.load("en_core_web_sm")


def clean_text(text: str) -> str:
    """Basic text cleaning: remove HTML tags, extra whitespace, control chars."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<.*?>", " ", text)          # strip HTML tags
    text = re.sub(r"\s+", " ", text)             # collapse whitespace
    text = text.strip()
    return text


def main():
    print(f"Loading dataset from {INPUT_PATH} ...")
    df = pd.read_csv(INPUT_PATH)
    print(f"Full dataset size: {len(df)} rows")

    # Keep only the columns we need, rename for consistency
    df = df[["Id", "ProductId", "Text", "Score"]].rename(
        columns={
            "Id": "review_id",
            "ProductId": "product_id",
            "Text": "original_review",
            "Score": "rating",
        }
    )

    # Drop rows with missing review text
    df = df.dropna(subset=["original_review"])

    # Sample a manageable subset
    if len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"Sampled dataset size: {len(df)} rows")

    # Clean text
    print("Cleaning review text...")
    df["original_review"] = df["original_review"].apply(clean_text)

    # Remove exact duplicate reviews
    before = len(df)
    df = df.drop_duplicates(subset=["original_review"])
    print(f"Removed {before - len(df)} duplicate reviews")

    # Split each review into sentences using SpaCy
    print("Splitting reviews into sentences (this may take a few minutes)...")
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        doc = nlp(row["original_review"])
        for sent in doc.sents:
            sentence_text = sent.text.strip()
            if len(sentence_text) < 3:
                continue  # skip empty/near-empty sentences
            rows.append(
                {
                    "review_id": row["review_id"],
                    "product_id": row["product_id"],
                    "sentence": sentence_text,
                    "original_review": row["original_review"],
                    "rating": row["rating"],
                }
            )

    output_df = pd.DataFrame(rows)
    print(f"Generated {len(output_df)} sentences from {len(df)} reviews")

    output_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved cleaned + sentence-split data to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
