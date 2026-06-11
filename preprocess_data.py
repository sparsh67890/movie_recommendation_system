import pandas as pd
import numpy as np
import os
import glob

def load_netflix_data(raw_dir):
    data = []
    # Kaggle dataset usually has combined_data_1.txt, etc.
    file_pattern = os.path.join(raw_dir, 'combined_data_*.txt')
    files = sorted(glob.glob(file_pattern))
    
    if not files:
        print(f"No data files found matching {file_pattern}.")
        print("Please ensure you have downloaded the Kaggle Netflix Prize dataset and extracted it into netflix_recommender/data/raw/")
        return pd.DataFrame()
        
    for file in files:
        print(f"Reading {os.path.basename(file)}...")
        with open(file, 'r') as f:
            movie_id = None
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.endswith(':'):
                    movie_id = int(line[:-1])
                else:
                    parts = line.split(',')
                    if len(parts) == 3:
                        data.append([movie_id, int(parts[0]), int(parts[1]), parts[2]])
    
    print("Converting to DataFrame...")
    df = pd.DataFrame(data, columns=['movie_id', 'user_id', 'rating', 'date'])
    return df

def main():
    raw_dir = 'netflix_recommender/data/raw'
    processed_dir = 'netflix_recommender/data/processed'
    os.makedirs(processed_dir, exist_ok=True)
    
    df = load_netflix_data(raw_dir)
    if df.empty:
        return
        
    print(f"Total ratings loaded: {len(df):,}")
    
    print("Filtering for Modeling Subset (Movies >= 500 ratings, Top 50,000 active users)...")
    # Movies with >= 500 ratings
    movie_counts = df['movie_id'].value_counts()
    valid_movies = movie_counts[movie_counts >= 500].index
    df_filtered = df[df['movie_id'].isin(valid_movies)]
    
    # 50,000 most active users
    user_counts = df_filtered['user_id'].value_counts()
    top_users = user_counts.head(50000).index
    modeling_subset = df_filtered[df_filtered['user_id'].isin(top_users)]
    
    modeling_path = os.path.join(processed_dir, 'netflix_modeling_subset.csv')
    print(f"Saving {modeling_path} ({len(modeling_subset):,} ratings)...")
    modeling_subset.to_csv(modeling_path, index=False)
    
    print("Creating SVD Subset (8000 random users)...")
    np.random.seed(42)
    random_users = np.random.choice(top_users, size=8000, replace=False)
    svd_subset = modeling_subset[modeling_subset['user_id'].isin(random_users)]
    
    svd_path = os.path.join(processed_dir, 'netflix_svd_subset.csv')
    print(f"Saving {svd_path} ({len(svd_subset):,} ratings)...")
    svd_subset.to_csv(svd_path, index=False)
    
    print("Generating legacy netflix_subset.csv (Users >= 50, Movies >= 100)...")
    # Users with >= 50 ratings
    u_counts = df['user_id'].value_counts()
    valid_u = u_counts[u_counts >= 50].index
    legacy_subset = df[df['user_id'].isin(valid_u)]
    # Movies with >= 100 ratings
    m_counts = legacy_subset['movie_id'].value_counts()
    valid_m = m_counts[m_counts >= 100].index
    legacy_subset = legacy_subset[legacy_subset['movie_id'].isin(valid_m)]
    
    legacy_path = os.path.join(processed_dir, 'netflix_subset.csv')
    print(f"Saving {legacy_path} ({len(legacy_subset):,} ratings)...")
    legacy_subset.to_csv(legacy_path, index=False)
    
    print("Data preprocessing complete! You can now safely delete the raw data if needed.")

if __name__ == "__main__":
    main()
