# Netflix Movie Recommendation System

## 1. Project Overview
This repository contains a complete, student-level machine learning project designed to solve the Netflix Prize dataset recommendation challenge. We built and evaluated a Hybrid Recommendation System that bridges the gap between pure predictive accuracy and transparent explainability, strictly adhering to the project's PDF guidelines.

## 2. Dataset Description
The project uses the Netflix Prize dataset. To ensure reproducibility on standard hardware, we processed the 100M+ ratings down into a highly dense **Modeling Subset** containing 43.2 million ratings across 50,000 active users and 7,291 movies. For the computationally heavy SVD and Item-CF evaluations, we sampled 8,000 users, resulting in an exact modeling dataset of **6.9 million ratings**.

## 3. Repository Structure
```
netflix_recommender/
├── app/                  # Streamlit Interactive Dashboard
├── data/                 # Raw and processed datasets (excluded from git due to size)
├── models/               # Serialized model artifacts (excluded from git due to size)
├── notebooks/            # Jupyter notebooks containing the entire ML lifecycle
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_item_based_cf.ipynb
│   ├── 03_svd.ipynb
│   ├── 04_hybrid_recommendation.ipynb
│   ├── 05_evaluation_metrics.ipynb
│   └── 06_recommendation_analysis.ipynb
└── reports/              # Auto-generated markdown components for the Final Technical Report
```

## 4. Data Processing Pipeline
1. Download the raw Kaggle dataset into `data/raw/`.
2. Run data filtering to extract active users (>=50 ratings) and popular movies (>=500 ratings).
3. Generate the 8,000 user sub-sample for SVD/KNN evaluation using a random seed for reproducible sampling.

## 5. Model Development Pipeline
- **Item-CF:** A traditional neighborhood model calculating Cosine Similarity across a sparse user-item matrix. It struggles with popularity bias but excels at providing explainable recommendations.
- **SVD (Singular Value Decomposition):** A matrix factorization technique (50 factors, 20 epochs) utilizing `scikit-surprise`. It acts as a black box but provides exceptionally low error rates.
- **Hybrid Model:** A weighted ensemble (0.7 SVD + 0.3 Item-CF) that utilizes Min-Max normalization to combine the SVD's mathematical accuracy with Item-CF's transparency.

## 6. Evaluation Methodology
Models were evaluated using an **80/20 train-test split**.
- **RMSE (Root Mean Squared Error):** Measured rating prediction accuracy.
- **MAP@10 (Mean Average Precision):** Measured ranking quality for Top-10 lists. We strictly enforced a relevance threshold of **>= 3.5**. 

## 7. Streamlit Dashboard Instructions
To interact with the models visually:
```bash
streamlit run netflix_recommender/app/main.py
```
*Note: The dashboard relies on serialized `.pkl` models generated in step 8. If they are missing, you must generate them first.*

## 8. Exact Reproduction Steps
Run all commands below from the project root directory.
### Run the project
### 1. Setup Environment
```bash
git clone <repository_url>
cd movie_recommendation_system

python3 -m venv venv
source venv/bin/activate

pip install -r netflix_recommender/requirements.txt
```

### 2. Preprocess Raw Data
Download the Netflix Prize dataset from Kaggle:

https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data

After downloading the dataset and extracting its contents (e.g., `combined_data_1.txt`, `combined_data_2.txt`, `combined_data_3.txt`, `combined_data_4.txt`, and `movie_titles.csv`) into: `netflix_recommender/data/raw/`

run the preprocessing script to generate the modeling subsets:

```bash
python preprocess_data.py
```

### 3. Serialize Models for Dashboard
```bash
python save_models.py
```

## 4. Run the Dashboard
```bash
streamlit run netflix_recommender/app/main.py
```
