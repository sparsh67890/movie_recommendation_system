import streamlit as st
import pickle
import os
from surprise import dump

@st.cache_resource
def load_models():
    """Load serialized SVD model and Item-CF artifacts."""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    
    # Load SVD
    svd_path = os.path.join(base_dir, 'models', 'svd_model.pkl')
    _, svd_model = dump.load(svd_path)
    
    # Load Item-CF artifacts
    cf_path = os.path.join(base_dir, 'models', 'item_cf_artifacts.pkl')
    with open(cf_path, 'rb') as f:
        cf_artifacts = pickle.load(f)
        
    return svd_model, cf_artifacts

@st.cache_data
def get_movie_list(_cf_artifacts):
    """Get sorted list of (movie_id, title) tuples for dropdowns."""
    title_map = _cf_artifacts['title_map']
    movies = [(m_id, title_map.get(m_id, f"Unknown (ID: {m_id})")) 
              for m_id in _cf_artifacts['movie_map'].values()]
    # Sort by title
    return sorted(movies, key=lambda x: str(x[1]))

@st.cache_data
def get_user_list(_cf_artifacts):
    """Get sorted list of user IDs for dropdowns."""
    return sorted(list(_cf_artifacts['user_map'].values()))
