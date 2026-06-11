# Recommendation Analysis & Discussion

## Qualitative Performance
While quantitative metrics (MAP@10 and RMSE) prove our Hybrid model is mathematically sound, we must also analyze its real-world behavior to understand its practical strengths and limitations.

### Success and Failure Cases
*(Note: The following scenarios are illustrative conceptual examples demonstrating the model's structural behavior.)*
- **Successes:** The Hybrid model excels when users have distinct, consistent taste clusters (e.g., exclusively rating 1980s horror films highly). The combination of latent factors and localized similarity accurately pinpoints niche titles within that specific subgenre.
- **Failures:** The model struggles with anomalous "guilty pleasure" behavior. If a user primarily enjoys high-brow dramas but occasionally gives 5 stars to a critically panned action movie out of ironic enjoyment, the model misinterprets this anomaly. It alters the latent profile and erroneously floods the user's recommendations with low-quality action films.

## Popularity Bias
Our analysis indicates a natural popularity bias in the dataset: the top 100 movies account for approximately 9% of all ratings. While not overwhelming, this concentration means pure Item-CF models can sometimes lean towards recommending familiar blockbusters. Our Hybrid model mitigates this by applying a 70% weight to the SVD predictions, helping the system balance this bias and uncover deeper latent affinities.

## The Cold-Start Limitation
The primary structural limitation of our system is the "Cold-Start" problem. Because Collaborative Filtering relies entirely on historical interactions:
1. **New Users:** If a user registers with zero ratings, the system defaults to outputting the global mean rating (~3.5 stars) for every movie.
2. **New Movies:** If a brand new movie is added to the catalog, it cannot be recommended because it lacks the interaction history required to calculate cosine similarity or latent factors.

## Future Improvements
To advance this system further, future iterations should incorporate **Content-Based Filtering**. By feeding movie metadata (genres, cast, plot summaries) into the algorithm alongside our collaborative filters, the system could recommend brand new movies based on their text attributes, entirely solving the Item Cold-Start problem.
