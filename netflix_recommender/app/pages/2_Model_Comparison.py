import streamlit as st
import pandas as pd

st.set_page_config(page_title="Model Comparison", page_icon="⚖️")

st.title("Model Comparison ⚖️")

st.markdown("""
We evaluated three different recommendation approaches strictly using an **80/20 train-test split**. 

The metrics below were calculated by comparing the models' predicted star ratings against the actual hidden ratings from the 20% test set.
""")

results = pd.DataFrame({
    'Model': ['Item-CF', 'SVD', 'Hybrid (0.7 SVD + 0.3 CF)'],
    'RMSE': [0.9421, 0.8013, 0.8136],
    'MAE': [0.7283, 0.6212, 0.6310],
    'MAP@10 (Relevance >= 3.5)': [0.7697, 0.8754, 0.8800]
})

st.table(results.style.format({
    'RMSE': '{:.4f}',
    'MAE': '{:.4f}',
    'MAP@10 (Relevance >= 3.5)': '{:.4f}'
}))

st.subheader("Conclusion")
st.markdown("""
- **SVD is best for rating prediction accuracy.** It acts as a powerful mathematical engine, minimizing the average magnitude of prediction errors (lowest RMSE and MAE).
- **Hybrid is best for recommendation ranking quality.** By injecting a 30% localized similarity signal into the SVD base, it slightly sacrifices pure numerical accuracy (RMSE rises to 0.8136, MAE to 0.6310) but ultimately generates the most relevant Top-10 lists (highest MAP@10).
- **Item-CF provides explainability.** While it has the worst standalone accuracy metrics, its transparent "users who liked X also liked Y" structure is what makes the Hybrid model trustworthy to end-users.
""")
