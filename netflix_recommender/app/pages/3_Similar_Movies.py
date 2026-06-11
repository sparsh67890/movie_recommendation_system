import streamlit as st
from utils import load_models, get_movie_list
import numpy as np

st.set_page_config(page_title="Similar Movies", page_icon="🔗")

st.title("Find Similar Movies 🔗")
st.markdown("Powered by our **Item-Based Collaborative Filtering** matrix.")

with st.spinner("Loading models..."):
    _, cf_artifacts = load_models()
    movie_list = get_movie_list(cf_artifacts)

# Select a movie
selected_movie = st.selectbox(
    "Select a movie you like:",
    options=movie_list,
    format_func=lambda x: x[1]
)

if st.button("Find Similar Movies"):
    movie_id = selected_movie[0]
    
    # Check if movie is in our mapping
    if movie_id in cf_artifacts['rev_movie_map']:
        internal_idx = cf_artifacts['rev_movie_map'][movie_id]
        
        # Get similarities
        sim_scores = cf_artifacts['item_similarity'][internal_idx]
        
        # Get top 10 indices
        top_indices = np.argsort(sim_scores)[::-1][:10]
        
        results = []
        for idx in top_indices:
            sim = sim_scores[idx]
            if sim > 0:
                rec_movie_id = cf_artifacts['movie_map'][idx]
                rec_title = cf_artifacts['title_map'].get(rec_movie_id, f"Unknown ({rec_movie_id})")
                results.append({"Movie": rec_title, "Similarity Score": round(sim, 4)})
        
        if results:
            st.table(results)
        else:
            st.warning("No similar movies found. The similarity matrix might be too sparse for this title.")
    else:
        st.error("Movie not found in the trained subset.")
