# Hybrid Recommendation System

## Objective
The goal of this phase was to construct a Hybrid Recommendation System that bridges the gap between accuracy and explainability. By combining Singular Value Decomposition (SVD) and Item-Based Collaborative Filtering (Item-CF), we aimed to leverage the mathematical strength of latent factors while still providing understandable and relatable recommendations to the user.

## Methodology
Instead of relying on a single algorithm, our system generates predictions using both models simultaneously. 
1. **Candidate Generation:** For a given user, the system identifies all unrated movies in the dataset.
2. **Scoring:** The system predicts scores for these movies using both the trained SVD model and the Item-CF similarity matrix.
3. **Normalization:** Because SVD outputs star ratings (1-5) and Item-CF outputs unbounded dot-product scores, we applied Min-Max normalization to scale both scores linearly between 0 and 1.
4. **Weighted Combination:** We calculated a final score using a weighted average.

## Hybrid Formula
Through experimentation, we determined that leaning primarily on SVD for predictive accuracy provides better overall results, while a smaller portion of Item-CF ensures the recommendations remain relatable. The final scoring equation used was:

**Final Score = (0.7 × Normalized SVD Score) + (0.3 × Normalized Item-CF Score)**

## Sample Results
When tested against sample users, the hybrid function successfully filtered out previously watched titles and returned a sorted DataFrame containing the `movie_id`, `title`, individual model scores, the `final_score`, and an `explanation` string, demonstrating that the function works effectively.

## Advantages
- **Balanced Output:** The Hybrid model successfully balances global predictive power (SVD) with local similarity matching (Item-CF). 
- **Explainable Results:** SVD alone acts as a "black box," but the 30% Item-CF injection guarantees that the final recommendations share direct similarities with the user's past viewing history. This allows us to confidently display an explanation such as: *"Recommended because similar users liked this movie and it is also similar to movies you previously rated highly."*
- **Bias Mitigation:** SVD helps temper the "popularity bias" inherent in Item-CF, preventing the system from just recommending the same blockbusters repeatedly.

## Limitations
- **Cold-Start Problems:** The Hybrid model still relies entirely on historical interactions. If a new user registers and provides zero ratings, both the Item-CF and SVD components fail to generate personalized scores, resulting in generic default recommendations.
- **Computational Overhead:** Scoring candidates against two separate models simultaneously doubles the inference time compared to using a single baseline model, making real-time generation more resource-intensive.
