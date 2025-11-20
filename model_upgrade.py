import pandas as pd
import numpy as np
import ast
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer

# --- 1. CLEAN & PREPROCESS TEXT ---
def convert_list(obj):
    """Extracts names from stringified lists."""
    L = []
    try:
        for i in ast.literal_eval(obj):
            L.append(i['name'])
    except (ValueError, SyntaxError):
        return []
    return L

def fetch_director(obj):
    """Extracts director name."""
    L = []
    try:
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                L.append(i['name'])
                break
    except (ValueError, SyntaxError):
        return []
    return L

def preprocess_data():
    print("Loading Data...")
    # Ensure these CSV files are in the same folder as this script
    movies = pd.read_csv('tmdb_5000_movies.csv')
    credits = pd.read_csv('tmdb_5000_credits.csv')

    movies = movies.merge(credits, on='title')
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew', 'vote_average', 'vote_count', 'popularity']]
    
    print("Cleaning Data...")
    movies.dropna(inplace=True)

    movies.reset_index(drop=True, inplace=True)

    movies['genres'] = movies['genres'].apply(convert_list)
    movies['keywords'] = movies['keywords'].apply(convert_list)
    movies['cast'] = movies['cast'].apply(lambda x: convert_list(x)[:3])
    movies['crew'] = movies['crew'].apply(fetch_director)

    def collapse(L):
        return [i.replace(" ", "") for i in L]

    movies['genres'] = movies['genres'].apply(collapse)
    movies['keywords'] = movies['keywords'].apply(collapse)
    movies['cast'] = movies['cast'].apply(collapse)
    movies['crew'] = movies['crew'].apply(collapse)
    
    movies['overview_list'] = movies['overview'].apply(lambda x: x.split())
    
    # Create unified tags for TF-IDF
    movies['tags'] = movies['overview_list'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    movies['tags_str'] = movies['tags'].apply(lambda x: " ".join(x).lower())

    # --- POPULARITY NORMALIZATION ---
    # This helps surface better quality movies
    movies['pop_score'] = np.log1p(movies['vote_count']) * movies['vote_average']
    scaler = MinMaxScaler()
    movies['normalized_pop'] = scaler.fit_transform(movies[['pop_score']])

    return movies

# --- 2. TF-IDF ---
def compute_tfidf(movies):
    print("Computing TF-IDF Matrix...")
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['tags_str'])
    return cosine_similarity(tfidf_matrix)

# --- 3. BERT EMBEDDINGS ---
def compute_bert(movies):
    print("Computing BERT Embeddings (Downloads model once, might take time)...")
    # This uses a lightweight but powerful model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(movies['overview'].tolist(), show_progress_bar=True)
    return cosine_similarity(embeddings)

# --- 4. HYBRID ENGINE ---
def compute_hybrid(tfidf_sim, bert_sim, movies):
    print("Calculating Hybrid Scores...")
    # Weights: 50% Context (BERT), 30% Keywords (TFIDF), 20% Popularity
    hybrid_sim = (0.5 * bert_sim) + (0.3 * tfidf_sim) + (0.2 * movies['normalized_pop'].values)
    return hybrid_sim

if __name__ == "__main__":
    # Execution Flow
    df = preprocess_data()
    tfidf_sim = compute_tfidf(df)
    bert_sim = compute_bert(df)
    final_sim = compute_hybrid(tfidf_sim, bert_sim, df)
    
    print("Saving files...")
    pickle.dump(df, open('movies.pkl', 'wb'))
    pickle.dump(final_sim, open('similarity.pkl', 'wb'))
    print("DONE! You can now run your Flask app.")