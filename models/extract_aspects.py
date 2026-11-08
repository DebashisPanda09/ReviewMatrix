"""
ReviewMatrix - Aspect Extraction Script
-----------------------------------------
Loads cleaned/sentence-split reviews and classifies each sentence against
a fixed set of product aspects using zero-shot classification (BART-MNLI).

Input:  data/processed/cleaned_reviews.csv
Output: data/processed/aspects.csv
        Columns: review_id, product_id, sentence, aspect, confidence, rating
"""

import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm

# ---- Config ----
INPUT_PATH = "data/processed/cleaned_reviews.csv"
OUTPUT_PATH = "data/processed/aspects.csv"
CONFIDENCE_THRESHOLD = 0.5
BATCH_SIZE = 8  # keep small for 6GB VRAM laptops

# Aspect taxonomy for food/grocery products (Amazon Fine Food Reviews)
ASPECTS = [
    "Taste",
    "Freshness",
    "Packaging",
    "Price",
    "Delivery Experience",
    "Quantity or Value",
]


def main():
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

    print("Loading zero-shot classification model (facebook/bart-large-mnli)...")
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=device,
    )

    print(f"Loading sentences from {INPUT_PATH} ...")
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} sentences")

    results = []
    sentences = df["sentence"].astype(str).tolist()

    print("Classifying sentences against aspect taxonomy...")
    for start in tqdm(range(0, len(sentences), BATCH_SIZE)):
        batch = sentences[start : start + BATCH_SIZE]
        batch_rows = df.iloc[start : start + BATCH_SIZE]

        outputs = classifier(batch, candidate_labels=ASPECTS, multi_label=True)
        # classifier returns a single dict if batch size 1, else a list
        if isinstance(outputs, dict):
            outputs = [outputs]

        for row, output in zip(batch_rows.itertuples(index=False), outputs):
            # Keep every aspect that clears the confidence threshold
            # (a sentence can legitimately touch more than one aspect)
            for label, score in zip(output["labels"], output["scores"]):
                if score >= CONFIDENCE_THRESHOLD:
                    results.append(
                        {
                            "review_id": row.review_id,
                            "product_id": row.product_id,
                            "sentence": row.sentence,
                            "aspect": label,
                            "confidence": round(float(score), 4),
                            "rating": row.rating,
                        }
                    )

    output_df = pd.DataFrame(results)
    print(f"Generated {len(output_df)} aspect-tagged sentence rows")
    print("Aspect distribution:")
    print(output_df["aspect"].value_counts())

    output_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved aspect-tagged data to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
