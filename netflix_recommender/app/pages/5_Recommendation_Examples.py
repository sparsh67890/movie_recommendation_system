import streamlit as st
import pandas as pd
from utils import load_models, get_user_list
import numpy as np

st.set_page_config(page_title="Recommendation Examples", page_icon="🌟")

st.title("Recommendation Examples 🌟")

st.markdown("""
This page presents several sample users and demonstrates exactly what the Hybrid Model recommends to them, alongside explainability text.
""")

with st.spinner("Loading models..."):
    svd_model, cf_artifacts = load_models()
    # We pick 3 random but consistent users to act as our examples
    user_list = get_user_list(cf_artifacts)
    sample_users = [user_list[10], user_list[50], user_list[100]] 

def generate_recommendations(selected_user):
    internal_user_idx = cf_artifacts['rev_user_map'][selected_user]
    user_row = cf_artifacts['sparse_user_item'].getrow(internal_user_idx)
    rated_internal_indices = set(user_row.indices)
    
    candidates = []
    
    for internal_movie_idx, movie_id in cf_artifacts['movie_map'].items():
        if internal_movie_idx in rated_internal_indices:
            continue
            
        svd_pred = svd_model.predict(selected_user, movie_id).est
        item_sims = cf_artifacts['item_similarity'][internal_movie_idx]
        user_ratings = user_row.toarray()[0]
        cf_raw = np.dot(item_sims, user_ratings)
        
        candidates.append({
            'movie_id': movie_id,
            'title': cf_artifacts['title_map'].get(movie_id, f"Unknown ({movie_id})"),
            'svd_score': svd_pred,
            'cf_raw': cf_raw,
            'internal_idx': internal_movie_idx
        })
        
    df = pd.DataFrame(candidates)
    min_cf = df['cf_raw'].min()
    max_cf = df['cf_raw'].max()
    
    if max_cf > min_cf:
        df['cf_norm'] = 1 + 4 * ((df['cf_raw'] - min_cf) / (max_cf - min_cf))
    else:
        df['cf_norm'] = 3.0
        
    df['hybrid_score'] = (0.7 * df['svd_score']) + (0.3 * df['cf_norm'])
    top_10 = df.sort_values(by='hybrid_score', ascending=False).head(10)
    
    return top_10, len(rated_internal_indices)

for i, user in enumerate(sample_users):
    st.subheader(f"Example {i+1}: User ID {user}")
    
    top_10, rated_count = generate_recommendations(user)
    
    st.markdown(f"**Context:** This user has previously rated **{rated_count}** movies in our dataset.")
    
    results = []
    for _, row in top_10.iterrows():
        results.append({
            "Recommended Movie": row['title'],
            "Hybrid Score": round(row['hybrid_score'], 2),
            "Explanation": f"Based on strong latent matching (SVD: {row['svd_score']:.1f}) and direct similarity to your previously highly-rated movies."
        })
        
    st.table(pd.DataFrame(results))
    st.divider()
