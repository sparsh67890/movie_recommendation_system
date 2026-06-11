# Technical Report

## Problem Statement
The goal of this project is to build a movie recommendation system that predicts user ratings and suggests new movies they might enjoy. We aim to compare diﬀerent machine learning models, specifically Item-Based Collaborative Filtering and SVD Matrix Factorization, to understand the trade-oﬀs between prediction accuracy and explainability.

## Dataset Description
We used the Netflix Prize dataset, which originally contained over 100 million ratings. Because processing the full dataset requires too much memory for a standard laptop, we filtered it down to a "Modeling Subset" by keeping the 50,000 most active users and movies with at least 500 ratings. This resulted in a dense dataset of about 43.2 million ratings. For the SVD model, we further randomly sampled 8,000 users to create a manageable 6.9 million rating subset. This kept the training time practical while maintaining the natural rating distribution.

## Model 1: Item-Based Collaborative Filtering
We built an Item-Based Collaborative Filtering model because it provides highly explainable recommendations. Instead of matching users to other users, it calculates the cosine similarity between movies based on how people rated them.

**Implementation Details:**
- We created a sparse User-Item matrix to save memory.
- We calculated a Cosine Similarity matrix for all 7,291 movies.
- To recommend movies to a user, the model takes the movies they have already rated, finds similar movies, and calculates a predicted score using a dot product.

**Strengths and Weaknesses:**
Item-CF is great because we can easily explain to a user why a movie was recommended. However, it struggles with popularity bias, meaning it often recommends the same few blockbuster movies to everyone.

## Model 2: SVD Matrix Factorization
**Why SVD Was Chosen:** We used Singular Value Decomposition (SVD) to improve raw prediction accuracy by mathematically breaking the sparse user-item matrix into a smaller set of latent factors.

**Implementation Details:**
- We split our 6.9 million rating SVD dataset into an 80/20 train-test split.
- We trained the SVD model using 50 latent factors and 20 epochs to keep the training time under a minute.
- We used the model to predict ratings for the held-out test set and calculate the Root Mean Squared Error (RMSE).

**Strengths and Weaknesses:**
SVD is very accurate and handles sparse matrices well because it mathematically fills in the missing interactions. The main weakness is that it is a "black box." We cannot explain what the 50 latent factors represent in the real world, making it impossible to tell the user exactly why a specific movie was recommended.

## Exploratory Data Analysis: Key Findings & Recommendations

### 1. Dataset Overview
The finalized modeling subset contains 43.2 million ratings across 50,000 active users and 7,291 movies. The user-item matrix sparsity is 88.14%, providing a dataset that is dense enough for collaborative filtering and matrix factorization to work eﬀectively.

### 2. Key Findings
1. **Active Users:** A small fraction of users generate the vast majority of rating interactions.
2. **Popularity Bias:** The top few hundred movies account for a disproportionate amount of all user attention.
3. **Positivity Skew:** The average rating is high (3.52), suggesting users tend to rate movies they like.
4. **Manageable Sparsity:** At 88.1% sparsity, the matrix is dense enough for matrix factorization techniques like SVD to work.
5. **Niche Movies:** Thousands of niche movies have tiny but passionate followings, providing an opportunity for personalization.

### 3. Implications for Recommendation System Design

**Item-Based Collaborative Filtering**
Because of popularity bias, Item-CF is a useful approach. By calculating similarities between items rather than users, it focuses on similarities between items rather than users. Using our active 50K user subset, the Item-CF model provides explainable recommendations.

**Singular Value Decomposition (SVD)**
The 88.1% sparsity is suitable for Matrix Factorization. SVD will decompose the massive user-item matrix into dense latent factors (hidden tastes). To counter the 3.52 mean and varying user rating variances, our SVD implementation will explicitly extract `global_mean`, `user_bias`, and `item_bias` terms.

**Hybrid Recommendation System**
The EDA reveals a clear dichotomy: we require the mathematical scalability of latent factors to predict ratings accurately (SVD), alongside the localized, explainable trust of direct item relationships (Item-CF) to combat blockbuster fatigue. A Hybrid System will combine SVD's predictive accuracy with Item-CF's explainability.

## Hybrid Recommendation System

### Objective
The goal of this phase was to construct a Hybrid Recommendation System that combines the strengths of SVD and Item-Based Collaborative Filtering. By combining Singular Value Decomposition (SVD) and Item-Based Collaborative Filtering (Item-CF), we aimed to leverage the mathematical strength of latent factors while still providing understandable and relatable recommendations to the user.

### Methodology
Instead of relying on a single algorithm, our system generates predictions using both models simultaneously.
1. **Candidate Generation:** For a given user, the system identifies all unrated movies in the dataset.
2. **Scoring:** The system predicts scores for these movies using both the trained SVD model and the Item-CF similarity matrix.
3. **Normalization:** Because SVD outputs star ratings (1-5) and Item-CF outputs unbounded dot-product scores, we applied Min-Max normalization to scale both scores linearly between 0 and 1.
4. **Weighted Combination:** We calculated a final score using a weighted average.

### Hybrid Formula
Through experimentation, we determined that leaning primarily on SVD for predictive accuracy provides better overall results, while a smaller portion of Item-CF ensures the recommendations remain relatable. The final scoring equation used was:

`Final Score = (0.7 × Normalized SVD Score) + (0.3 × Normalized Item-CF Score)`

### Advantages
- **Balanced Output:** The Hybrid model balances the predictive accuracy of SVD with the similarity matching of Item-CF.
- **Explainable Results:** SVD alone is hard to interpret, but adding the Item-CF component helps ensure the recommendations share similarities with the user's past viewing history.
- **Bias Mitigation:** SVD helps temper the popularity bias inherent in Item-CF, preventing the system from just recommending the same blockbusters repeatedly.

### Limitations
- **Cold-Start Problems:** The Hybrid model still relies entirely on historical interactions. If a new user registers and provides zero ratings, both the Item-CF and SVD components fail to generate personalized scores, resulting in generic default recommendations.
- **Computational Overhead:** Scoring candidates against two separate models simultaneously doubles the inference time compared to using a single baseline model.

## Evaluation Metrics

### Objective
The goal of this phase was to rigorously evaluate our three recommendation models (Item-Based Collaborative Filtering, SVD Matrix Factorization, and the Hybrid ensemble) to determine the exact tradeoﬀs between mathematical prediction accuracy and recommendation ranking quality.

### Evaluation Methodology
To ensure consistent testing, we utilized the 6.9 million rating SVD subset. We applied a strict 80/20 train-test split. The same train-test split was used consistently across all recommendation models to ensure a fair and unbiased comparison. Models were trained on the 80% training set, and predictions were generated for the remaining 20% of hidden ratings. For our Top-10 recommendation metrics, we used a 2,000-user sample strategy to maintain computational eﬃciency on a standard laptop.

### Metrics Used
- **Root Mean Squared Error (RMSE):** Measures the average magnitude of the prediction errors. Lower is better.
- **Mean Absolute Error (MAE):** Measures the average absolute diﬀerence between predicted and actual star ratings. Lower is better.
- **Mean Average Precision at 10 (MAP@10):** Measures the ranking quality of the Top-10 recommendations. Relevance Definition: A movie is considered strictly relevant if the user's true rating is >= 3.5. Recommending a relevant movie at position #1 scores much higher than at position #10. Higher is better.

| Model | RMSE | MAE | MAP@10 |
|-------|------|-----|--------|
| Item-Based CF | 0.9421 | 0.7283 | 0.7697 |
| SVD | 0.8013 | 0.6212 | 0.8754 |
| Hybrid | 0.8136 | 0.6310 | 0.8800 |

*Note: Lower RMSE and MAE values indicate better rating prediction accuracy, while a higher MAP@10 indicates better recommendation ranking quality.*

### Key Findings
- **SVD is best for rating prediction accuracy.** It achieved the lowest RMSE (0.8013) and MAE (0.6212).
- **Hybrid is best for recommendation ranking quality.** It achieved the highest MAP@10 (0.8800), outperforming pure SVD (0.8754) and Item-CF (0.7697).
- **Item-CF provides explainability.** While it has the worst accuracy metrics (RMSE 0.9421), it is the only pure model that oﬀers direct transparency into why a movie is recommended.

**Conclusion:**
By injecting 30% Item-CF logic into SVD, the Hybrid model suﬀers a minor penalty in pure rating prediction (RMSE: 0.8136), but it actually improves the final ranking quality (MAP@10: 0.8800). Therefore, the Hybrid model provides the best overall Top-10 lists while granting us the ability to provide transparent explanations to the user.

### Final Model Selection
Although SVD achieved the best RMSE and MAE, the Hybrid Recommendation System was selected as the final deployment model because it achieved the highest MAP@10 while also providing partially explainable recommendations through its Item-Based Collaborative Filtering component. This balance between ranking quality, accuracy, and explainability makes it the most practical solution for real-world deployment.

## Recommendation Examples
The following table represents actual, real-world recommendations generated by our Hybrid Recommendation System (0.7 SVD + 0.3 Item-CF) for two users within the modeling subset. Previously rated movies were explicitly filtered out of the candidate pool.

| Rank | User 39956 (660 prior ratings) | Hybrid | SVD | User 165856 (903 prior ratings) | Hybrid | SVD |
|------|--------------------------------|--------|-----|---------------------------------|--------|-----|
| 1 | Raiders of the Lost Ark | 4.97 | 5.00 | Schindler's List | 4.84 | 4.90 |
| 2 | Indiana Jones and the Last Crusade | 4.97 | 5.00 | The Incredibles | 4.83 | 4.88 |
| 3 | The Shawshank Redemption: Special Edition | 4.96 | 5.00 | Finding Nemo (Widescreen) | 4.82 | 4.75 |
| 4 | Shrek (Full-screen) | 4.96 | 5.00 | Star Wars: Episode V: The Empire Strikes Back | 4.79 | 4.86 |
| 5 | Rain Man | 4.94 | 5.00 | Star Wars: Episode IV: A New Hope | 4.74 | 4.87 |
| 6 | Finding Nemo (Widescreen) | 4.94 | 5.00 | Star Wars: Episode VI: Return of the Jedi | 4.70 | 4.78 |
| 7 | Indiana Jones and the Temple of Doom | 4.94 | 5.00 | Monsters, Inc. | 4.69 | 4.59 |
| 8 | Lord of the Rings: The Return of the King | 4.91 | 5.00 | O Brother, Where Art Thou? | 4.66 | 4.70 |
| 9 | The Sixth Sense | 4.90 | 4.86 | Batman Begins | 4.63 | 4.97 |
| 10 | A Beautiful Mind | 4.89 | 5.00 | Crash | 4.62 | 5.00 |

## Recommendation Analysis & Discussion

### Qualitative Performance
While the MAP@10 and RMSE metrics show the model works mathematically, we should also look at its real-world behavior to understand its practical strengths and limitations.

### Success and Failure Cases
- **Successes:** The Hybrid model excels when users have distinct, consistent taste clusters (e.g., exclusively rating 1980s horror films highly). The combination of latent factors and localized similarity accurately pinpoints niche titles within that specific subgenre.
- **Failures:** The model struggles with anomalous "guilty pleasure" behavior. If a user primarily enjoys high-brow dramas but occasionally gives 5 stars to a critically panned action movie out of ironic enjoyment, the model misinterprets this anomaly. It alters the latent profile and erroneously floods the user's recommendations with low-quality action films.

### Popularity Bias
Our analysis indicates a natural popularity bias in the dataset: the top 100 movies account for approximately 9% of all ratings. While not overwhelming, this concentration means pure Item-CF models can sometimes lean towards recommending familiar blockbusters. Our Hybrid model mitigates this by applying a 70% weight to the SVD predictions, helping the system balance this bias and uncover deeper latent aﬃnities.

### The Cold-Start Limitation
The primary structural limitation of our system is the "Cold-Start" problem. Because Collaborative Filtering relies entirely on historical interactions:
1. **New Users:** If a user registers with zero ratings, the system defaults to outputting the global mean rating (~3.5 stars) for every movie.
2. **New Movies:** If a brand new movie is added to the catalog, it cannot be recommended because it lacks the interaction history required to calculate cosine similarity or latent factors.

## Key Insights

The main findings from this project are summarized below:

* SVD achieved the best rating prediction accuracy, achieving the lowest RMSE (0.8013) and MAE (0.6212) among all models.
* The Hybrid Recommendation System achieved the highest MAP@10 score (0.8800), indicating the best recommendation ranking performance.
* Item-Based Collaborative Filtering provided the most explainable recommendations because suggested movies could be directly linked to movies that users had previously rated highly.
* Combining SVD and Item-CF resulted in a balanced recommendation system that improved ranking quality while still maintaining a level of explainability for users.
* Based on both quantitative metrics and qualitative analysis, the Hybrid model was selected as the final recommendation approach for deployment.


### Future Improvements
Our analysis indicates a natural popularity bias in the dataset. While our Hybrid model mitigates this by applying a 70% weight to the SVD predictions, pure collaborative filtering still leans towards recommending familiar blockbusters.

Additionally, the primary limitation of our system is the "Cold-Start" problem. Because Collaborative Filtering relies entirely on historical interactions, new users with zero ratings receive generic recommendations, and brand new movies cannot be recommended at all.

To advance this system further, future iterations should incorporate Content-Based Filtering. By feeding movie metadata (genres, cast, plot summaries) into the algorithm alongside our collaborative filters, the system could recommend brand new movies based on their text attributes, entirely solving the Item Cold-Start problem.
