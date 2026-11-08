"""
ReviewMatrix - Sentiment Scoring Script
------------------------------------------
Loads aspect-tagged sentences and scores the sentiment of each one using
a pretrained RoBERTa sentiment model.

Input:  data/processed/aspects.csv
Output: data/processed/aspect_sentiments.csv
        Columns: review_id, product_id, sentence, aspect, sentiment,
                 sentiment_confidence, aspect_confidence, rating
"""

import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm

# ---- Config ----
INPUT_PATH = "data/processed/aspects.csv"
OUTPUT_PATH = "data/processed/aspect_sentiments.csv"
BATCH_SIZE = 16  # sentiment model is lighter than zero-shot, can go higher

# Maps the model's raw output labels to readable sentiment labels
LABEL_MAP = {
    "positive": "Positive",
    "negative": "Negative",
    "neutral": "Neutral",
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive",
}


def main():
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

    print("Loading sentiment model (cardiffnlp/twitter-roberta-base-sentiment-latest)...")
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=device,
    )

    print(f"Loading aspect-tagged sentences from {INPUT_PATH} ...")
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} aspect-tagged rows")

    results = []
    sentences = df["sentence"].astype(str).tolist()

    print("Scoring sentiment for each sentence...")
    for start in tqdm(range(0, len(sentences), BATCH_SIZE)):
        batch = sentences[start : start + BATCH_SIZE]
        batch_rows = df.iloc[start : start + BATCH_SIZE]

        # truncation=True guards against any unusually long sentences
        outputs = sentiment_pipe(batch, truncation=True)

        for row, output in zip(batch_rows.itertuples(index=False), outputs):
            raw_label = output["label"]
            sentiment = LABEL_MAP.get(raw_label, raw_label)
            results.append(
                {
                    "review_id": row.review_id,
                    "product_id": row.product_id,
                    "sentence": row.sentence,
                    "aspect": row.aspect,
                    "sentiment": sentiment,
                    "sentiment_confidence": round(float(output["score"]), 4),
                    "aspect_confidence": row.confidence,
                    "rating": row.rating,
                }
            )

    output_df = pd.DataFrame(results)
    print(f"Generated {len(output_df)} aspect-sentiment rows")
    print("\nSentiment distribution overall:")
    print(output_df["sentiment"].value_counts())
    print("\nSentiment distribution by aspect:")
    print(output_df.groupby(["aspect", "sentiment"]).size())

    output_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved aspect-sentiment data to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
