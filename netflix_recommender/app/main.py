import streamlit as st

st.set_page_config(
    page_title="Netflix Recommender App",
    page_icon="🎬",
    layout="wide",
)

st.title("Netflix Movie Recommendation System 🎬")

st.markdown("""
Welcome to the Netflix Recommendation Dashboard!

This project was built as an undergraduate machine learning implementation to explore collaborative filtering.

### Navigation
Use the sidebar on the left to explore:
- **1. Dataset Insights**: Understand the structure and biases of our modeling subset.
- **2. Model Comparison**: Review the rigorous RMSE and MAP@10 tradeoffs between SVD, Item-CF, and Hybrid approaches.
- **3. Similar Movies**: Provide a movie and see what the Item-CF matrix recommends.
- **4. User Recommendations**: See the fully personalized Hybrid recommendations for a specific user.
- **5. Recommendation Examples**: See pre-calculated examples of the system in action with explainability text.
""")
