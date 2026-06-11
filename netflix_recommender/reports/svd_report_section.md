# Model 2: SVD Matrix Factorization

## Why SVD Was Chosen
While our Item-Based Collaborative Filtering model provides excellent explainability, neighborhood models often struggle when the data is sparse. To improve our raw prediction accuracy, we implemented Singular Value Decomposition (SVD), a matrix factorization technique. SVD solves the sparsity problem by breaking down the large, mostly empty user-item matrix into a smaller, dense set of "latent factors" (hidden tastes or categories). By matching a user's latent profile against a movie's latent profile, SVD can mathematically predict ratings even when there is very little direct overlap between users.

## Dataset Used
Training matrix factorization on 43 million ratings requires significant memory and time. To ensure the model could be trained practically on our hardware, we created a dedicated SVD subset. Starting with our active modeling dataset, we randomly sampled 8,000 unique users. This gave us a final dataset of **6,924,035 ratings** across 7,291 movies, with a sparsity of 88.13%. By sampling entire user profiles rather than individual rows, we successfully reduced the file size to 184 MB while perfectly preserving the natural rating distribution and profile density.

## Train-Test Split
To evaluate how well the model generalizes to unseen data, we split the 6.9 million ratings into an **80/20 train-test split**. We used a fixed random seed (`random_state=42`) for reproducibility. This provided approximately 5.5 million ratings for the SVD algorithm to learn the latent factors, leaving 1.4 million hidden ratings to test the model's true predictive accuracy.

## Model Parameters
We built the model using the `scikit-surprise` Python library, which is heavily optimized for recommendation systems. The SVD algorithm was initialized with the following practical parameters:
- `n_factors = 50`: We reduced the number of latent factors from the default 100 down to 50. This significantly sped up the training time while still capturing enough variance in the data to make accurate predictions.
- `n_epochs = 20`: The model performed 20 iterations of stochastic gradient descent to minimize the prediction error.
- `random_state = 42`: Set to ensure our results are reproducible across different runs.

## RMSE Result
After training, we evaluated the model against our held-out test set. The model achieved a **Root Mean Squared Error (RMSE) of 0.8013**. This indicates that, on average, the SVD model's predicted rating was within 0.8 stars of the user's actual rating. This is a strong baseline performance for an undergraduate project and proves that the 50 latent factors successfully mapped the underlying user preferences.

## Strengths
The primary strength of SVD is its high predictive accuracy. Because the algorithm mathematically optimizes for the lowest possible global error, it generally outperforms Item-CF on pure prediction metrics like RMSE. Furthermore, it handles data sparsity exceptionally well because the factorization process naturally fills in the gaps through the dense latent dimensions.

## Limitations
The main limitation of SVD is its lack of interpretability—it acts as a "black box." The 50 latent factors it discovers are purely mathematical arrays. We cannot easily translate them into human-readable concepts (e.g., we cannot prove that Factor 12 represents "90s action movies"). Because of this, SVD cannot provide transparent, understandable explanations to the user about *why* a specific movie was recommended. Additionally, recalculating the matrix factors when new users or movies are added is computationally expensive.
