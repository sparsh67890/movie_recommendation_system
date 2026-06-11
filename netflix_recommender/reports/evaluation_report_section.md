# Evaluation Metrics

## Objective
The goal of this phase was to rigorously evaluate our three recommendation models (Item-Based Collaborative Filtering, SVD Matrix Factorization, and the Hybrid ensemble) to determine the exact tradeoffs between mathematical prediction accuracy and recommendation ranking quality.

## Evaluation Methodology
To ensure consistent testing, we utilized the 6.9 million rating SVD subset. We applied a strict **80/20 train-test split**. The same train-test split was used consistently across all recommendation models to ensure a fair and unbiased comparison. Models were trained on the 80% training set, and predictions were generated for the remaining 20% of hidden ratings. For our Top-10 recommendation metrics, we used a **2,000-user sample** strategy to maintain computational efficiency on a standard laptop.

## Metrics Used
- **Root Mean Squared Error (RMSE):** Measures the average magnitude of the prediction errors. Lower is better.
- **Mean Absolute Error (MAE):** Measures the average absolute difference between predicted and actual star ratings. Lower is better.
- **Mean Average Precision at 10 (MAP@10):** Measures the ranking quality of the Top-10 recommendations. **Relevance Definition:** A movie is considered strictly relevant if the user's true rating is **>= 3.5**. Recommending a relevant movie at position #1 scores much higher than at position #10. Higher is better.

| Model         | RMSE   | MAE    | MAP@10 |
| ------------- | ------ | ------ | ------ |
| Item-Based CF | 0.9421 | 0.7283 | 0.7697 |
| SVD           | 0.8013 | 0.6212 | 0.8754 |
| Hybrid        | 0.8136 | 0.6310 | 0.8800 |

*Note: Lower RMSE and MAE values indicate better rating prediction accuracy, while a higher MAP@10 indicates better recommendation ranking quality.*

## Key Findings
- **SVD is best for rating prediction accuracy.** It achieved the lowest RMSE (0.8013) and MAE (0.6212).
- **Hybrid is best for recommendation ranking quality.** It achieved the highest MAP@10 (0.8800), outperforming pure SVD (0.8754) and Item-CF (0.7697).
- **Item-CF provides explainability.** While it has the worst accuracy metrics (RMSE 0.9421), it is the only pure model that offers direct transparency into *why* a movie is recommended.

**Conclusion:** 
By injecting 30% Item-CF logic into SVD, the Hybrid model suffers a minor penalty in pure rating prediction (RMSE: 0.8136), but it actually *improves* the final ranking quality (MAP@10: 0.8800). Therefore, the Hybrid model provides the best overall Top-10 lists while granting us the ability to provide transparent explanations to the user.

## Final Model Selection
Although SVD achieved the best RMSE and MAE, the Hybrid Recommendation System was selected as the final deployment model because it achieved the highest MAP@10 while also providing partially explainable recommendations through its Item-Based Collaborative Filtering component. This balance between ranking quality, accuracy, and explainability makes it the most practical solution for real-world deployment.
