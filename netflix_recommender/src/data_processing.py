import os
import pandas as pd
import numpy as np

def load_data(raw_dir: str) -> pd.DataFrame:
    """
    Parses Netflix combined data files memory-efficiently.
    The format is:
    MovieID:
    CustomerID,Rating,Date
    """
    dfs = []
    
    files = [f for f in os.listdir(raw_dir) if f.startswith('combined_data_') and f.endswith('.txt')]
    for file in sorted(files):
        print(f"Processing {file}...")
        file_path = os.path.join(raw_dir, file)
        
        user_ids = []
        movie_ids = []
        ratings = []
        dates = []
        
        with open(file_path, 'r') as f:
            movie_id = None
            for line in f:
                line = line.strip()
                if line.endswith(':'):
                    movie_id = int(line[:-1])
                else:
                    parts = line.split(',')
                    if len(parts) == 3:
                        user_ids.append(int(parts[0]))
                        movie_ids.append(movie_id)
                        ratings.append(int(parts[1]))
                        dates.append(parts[2])
                        
        df_chunk = pd.DataFrame({
            'user_id': pd.Series(user_ids, dtype=np.int32),
            'movie_id': pd.Series(movie_ids, dtype=np.int32),
            'rating': pd.Series(ratings, dtype=np.float32),
            'date': pd.to_datetime(pd.Series(dates))
        })
        dfs.append(df_chunk)
        
    print("Concatenating DataFrames...")
    df = pd.concat(dfs, ignore_index=True)
    return df

def process_data(input_dir: str, output_file: str):
    """
    Loads data and applies rule-based filtering:
    - Users with >= 50 ratings
    - Movies with >= 100 ratings
    """
    df = load_data(input_dir)
    print(f"Original dataset size: {len(df)} ratings")
    
    # Filter users with >= 50 ratings
    user_counts = df['user_id'].value_counts()
    valid_users = user_counts[user_counts >= 50].index
    df = df[df['user_id'].isin(valid_users)]
    print(f"Dataset size after user filtering: {len(df)} ratings")
    
    # Filter movies with >= 100 ratings
    movie_counts = df['movie_id'].value_counts()
    valid_movies = movie_counts[movie_counts >= 100].index
    df = df[df['movie_id'].isin(valid_movies)]
    print(f"Dataset size after movie filtering: {len(df)} ratings")
    
    # Save to CSV
    print(f"Saving processed subset to {output_file}...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print("Processing complete!")

if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_dir = os.path.join(root_dir, "data", "raw")
    processed_data_file = os.path.join(root_dir, "data", "processed", "netflix_subset.csv")
    
    if not os.path.exists(processed_data_file):
        process_data(raw_data_dir, processed_data_file)
    else:
        print(f"Processed dataset already exists at {processed_data_file}")
