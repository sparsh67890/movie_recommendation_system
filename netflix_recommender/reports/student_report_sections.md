# Final Report Draft (Sections Completed So Far)

## Problem Statement
The goal of this project is to build a movie recommendation system that predicts user ratings and suggests new movies they might enjoy. We aim to compare different machine learning models, specifically Item-Based Collaborative Filtering and SVD Matrix Factorization, to understand the trade-offs between prediction accuracy and explainability.

## Dataset Description
We used the Netflix Prize dataset, which originally contained over 100 million ratings. Because processing the full dataset requires too much memory for a standard laptop, we filtered it down to a "Modeling Subset" by keeping the 50,000 most active users and movies with at least 500 ratings. This resulted in a dense dataset of about 43.2 million ratings. For the SVD model, we further randomly sampled 8,000 users to create a manageable 6.9 million rating subset. This kept the training time practical while perfectly preserving the natural rating distribution.

## Exploratory Data Analysis
During our EDA, we observed a few key trends in the dataset:
- **Positivity Bias:** Users tend to rate movies they like. The average rating is high (3.52 out of 5), and 4-star ratings are the most common.
- **Popularity Bias:** A small number of blockbuster movies receive the vast majority of ratings, while most movies get very few ratings (a classic long-tail distribution).
- **Data Sparsity:** Even after filtering for active users, 88.1% of the user-item matrix is empty because no single user has watched every movie. This sparsity is the main challenge our models need to overcome.

## Model 1: Item-Based Collaborative Filtering
We built an Item-Based Collaborative Filtering model because it provides highly explainable recommendations. Instead of matching users to other users, it calculates the cosine similarity between movies based on how people rated them. 

**Implementation Details:**
- We created a sparse User-Item matrix to save memory.
- We calculated a Cosine Similarity matrix for all 7,291 movies.
- To recommend movies to a user, the model takes the movies they have already rated, finds similar movies, and calculates a predicted score using a dot product.

**Strengths and Weaknesses:**
Item-CF is great because we can easily explain to a user why a movie was recommended (e.g., "Because you liked Movie A, we recommend Movie B"). However, it struggles with popularity bias, meaning it often recommends the same few blockbuster movies to everyone.

## Model 2: SVD Matrix Factorization
To improve mathematical prediction accuracy, we implemented Singular Value Decomposition (SVD) using the `scikit-surprise` library. SVD handles missing data (sparsity) much better than Item-CF by finding hidden "latent factors" that represent broad user tastes.

**Implementation Details:**
- We split our 6.9 million rating SVD dataset into 80% for training and 20% for testing.
- We trained the SVD model using 50 latent factors and 20 epochs to keep the training time under a minute.
- We used the model to predict ratings for the held-out test set and calculate the Root Mean Squared Error (RMSE).

**Strengths and Weaknesses:**
SVD is very accurate and handles sparse matrices well because it mathematically fills in the missing interactions. The main weakness is that it is a "black box." We cannot explain what the 50 latent factors represent in the real world, making it impossible to tell the user exactly why a specific movie was recommended.
