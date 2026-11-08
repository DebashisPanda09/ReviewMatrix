-- ReviewMatrix Database Schema
-- Run this once to set up the tables in the `reviewmatrix` database

DROP TABLE IF EXISTS aspect_sentiments;
DROP TABLE IF EXISTS reviews;

CREATE TABLE reviews (
    review_id BIGINT PRIMARY KEY,
    product_id TEXT NOT NULL,
    rating INT
);

CREATE TABLE aspect_sentiments (
    id SERIAL PRIMARY KEY,
    review_id BIGINT REFERENCES reviews(review_id),
    product_id TEXT NOT NULL,
    sentence TEXT NOT NULL,
    aspect TEXT NOT NULL,
    sentiment TEXT NOT NULL,
    sentiment_confidence FLOAT,
    aspect_confidence FLOAT
);

-- Indexes to speed up aggregation queries later
CREATE INDEX idx_aspect_sentiments_product ON aspect_sentiments(product_id);
CREATE INDEX idx_aspect_sentiments_aspect ON aspect_sentiments(aspect);
CREATE INDEX idx_aspect_sentiments_sentiment ON aspect_sentiments(sentiment);
