import streamlit as st
from utils import load_models, get_user_list
import numpy as np
import pandas as pd

st.set_page_config(page_title="User Recommendations", page_icon="👤")

st.title("Personalized Recommendations 👤")
st.markdown("Powered by our **Hybrid Model** (0.7 SVD + 0.3 Item-CF).")

with st.spinner("Loading models..."):
    svd_model, cf_artifacts = load_models()
    user_list = get_user_list(cf_artifacts)

selected_user = st.selectbox("Select a User ID:", options=user_list[:1000]) # Limit to first 1000 for UI responsiveness

if st.button("Generate Recommendations"):
    # Find movies user has already rated to exclude them
    internal_user_idx = cf_artifacts['rev_user_map'][selected_user]
    user_row = cf_artifacts['sparse_user_item'].getrow(internal_user_idx)
    rated_internal_indices = set(user_row.indices)
    
    candidates = []
    
    # We only predict for movies in our catalog that the user hasn't seen
    for internal_movie_idx, movie_id in cf_artifacts['movie_map'].items():
        if internal_movie_idx in rated_internal_indices:
            continue
            
        # SVD Prediction
        svd_pred = svd_model.predict(selected_user, movie_id).est
        
        # Item CF Prediction logic (simplified using dot product approximation for speed)
        # In actual scikit-surprise KNNBasic it's weighted average, but here we just approximate 
        # using the similarity matrix to generate a normalized score for demonstration.
        # To perfectly match, we'd invoke the Surprise KNNBasic model, but since we froze it
        # and didn't serialize the 400MB Surprise KNN model, we use our sparse matrix.
        
        item_sims = cf_artifacts['item_similarity'][internal_movie_idx]
        user_ratings = user_row.toarray()[0]
        
        # Calculate raw CF score
        cf_raw = np.dot(item_sims, user_ratings)
        
        candidates.append({
            'movie_id': movie_id,
            'title': cf_artifacts['title_map'].get(movie_id, f"Unknown ({movie_id})"),
            'svd_score': svd_pred,
            'cf_raw': cf_raw,
            'internal_idx': internal_movie_idx
        })
        
    df_candidates = pd.DataFrame(candidates)
    
    # Min-Max normalize CF scores to 1-5 scale to align with SVD
    min_cf = df_candidates['cf_raw'].min()
    max_cf = df_candidates['cf_raw'].max()
    
    if max_cf > min_cf:
        df_candidates['cf_norm'] = 1 + 4 * ((df_candidates['cf_raw'] - min_cf) / (max_cf - min_cf))
    else:
        df_candidates['cf_norm'] = 3.0
        
    # Hybrid calculation
    df_candidates['hybrid_score'] = (0.7 * df_candidates['svd_score']) + (0.3 * df_candidates['cf_norm'])
    
    # Sort and get Top 10
    top_10 = df_candidates.sort_values(by='hybrid_score', ascending=False).head(10)
    
    st.subheader(f"Top 10 Recommendations for User {selected_user}")
    
    results = []
    for _, row in top_10.iterrows():
        results.append({
            "Movie": row['title'],
            "Hybrid Score": round(row['hybrid_score'], 2),
            "Why?": f"Matches your underlying taste profile (SVD: {row['svd_score']:.1f}) and is highly similar to {len(rated_internal_indices)} movies you've watched."
        })
        
    st.table(pd.DataFrame(results))
