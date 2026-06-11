import streamlit as st

st.set_page_config(page_title="Dataset Insights", page_icon="📊")

st.title("Dataset Insights 📊")
st.subheader("Modeling Subset Statistics")

st.markdown("""
To ensure this application remains lightweight and reproducible on a standard laptop, we extracted a highly dense, representative subset from the original 100M Netflix Prize dataset.
""")

col1, col2, col3 = st.columns(3)
col1.metric("Total Ratings", "6.9 Million")
col2.metric("Total Users", "8,000")
col3.metric("Total Movies", "7,291")

st.divider()

st.subheader("Popularity Bias")
st.markdown("""
Our exploratory analysis indicated a natural **popularity bias** within this subset:
* **The top 100 movies** account for approximately **9%** of all 6.9 million ratings.

**Impact on Modeling:**
While not overwhelmingly massive, this 9% concentration means that pure localized models (like Item-CF) will sometimes lean too heavily toward recommending familiar blockbusters simply because they share a high volume of interaction overlaps with other movies. 

This mathematical reality is precisely why our **Hybrid Model** utilizes a 70% weight on the SVD matrix factorization—to counterbalance the blockbusters and uncover deeper, more personalized latent affinities.
""")
