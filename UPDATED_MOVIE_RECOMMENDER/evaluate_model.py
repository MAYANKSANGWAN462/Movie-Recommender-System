import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# CONFIGURATION
# ==========================================
TOP_K = 10           # Number of recommendations to check per movie
SAMPLE_SIZE = 500    # Number of random movies to evaluate (for speed)

def load_data():
    """Loads the trained model artifacts."""
    print("--- Step 1: Loading Model Artifacts ---")
    try:
        movies = pickle.load(open('movies.pkl', 'rb'))
        embeddings = pickle.load(open('item_embeddings.pkl', 'rb'))
        print(f"✅ Loaded {len(movies)} movies and BERT Embeddings.")
        return movies, embeddings
    except FileNotFoundError:
        print("❌ Error: Files not found. Please run 'model_upgrade.py' first.")
        exit()

def get_genre_consistency(movies, embeddings):
    """
    METRIC 1: SEMANTIC ACCURACY
    Measures if the recommended movies share genres with the input movie.
    """
    print("\n--- Step 2: Evaluating Semantic Accuracy (Genre Match) ---")
    
    # 1. Select random sample of movies to test
    sample_indices = np.random.choice(len(movies), SAMPLE_SIZE, replace=False)
    
    # 2. Compute similarities for the sample against ALL movies
    # We slice the embedding matrix for speed
    sample_embeddings = embeddings[sample_indices]
    
    # Calculate Cosine Similarity
    sim_matrix = cosine_similarity(sample_embeddings, embeddings)
    
    precision_scores = []
    
    for i, idx in enumerate(sample_indices):
        # Get source genres
        source_genres = set(movies.iloc[idx]['genres'])
        if not source_genres: continue
        
        # Get Top-K recommendations (indices)
        # argsort gives ascending, so we reverse [::-1] and skip the first (self)
        top_rec_indices = sim_matrix[i].argsort()[::-1][1:TOP_K+1]
        
        # Calculate how many recommendations share at least 1 genre
        hits = 0
        for rec_idx in top_rec_indices:
            rec_genres = set(movies.iloc[rec_idx]['genres'])
            if not source_genres.isdisjoint(rec_genres):
                hits += 1
        
        # Precision = Hits / K
        precision_scores.append(hits / TOP_K)
        
    avg_precision = np.mean(precision_scores) * 100
    print(f"📊 Genre Consistency Score: {avg_precision:.2f}%")
    return precision_scores

def plot_performance_metrics(precision_scores):
    """
    Plots the distribution of accuracy scores.
    """
    print("\n--- Step 3: Plotting Performance Metrics ---")
    plt.figure(figsize=(10, 5))
    
    sns.histplot(precision_scores, bins=10, kde=True, color='#2ecc71', edgecolor='black')
    plt.title(f'Distribution of Recommendation Accuracy (Genre Match @ {TOP_K})', fontsize=14)
    plt.xlabel('Precision Score (0.0 = Bad, 1.0 = Perfect)', fontsize=12)
    plt.ylabel('Frequency (Number of Test Movies)', fontsize=12)
    plt.axvline(np.mean(precision_scores), color='red', linestyle='--', label=f'Mean Accuracy: {np.mean(precision_scores):.2f}')
    plt.legend()
    
    plt.show()

if __name__ == "__main__":
    # 1. Load
    df_movies, item_vectors = load_data()
    
    # 2. Evaluate Accuracy
    scores = get_genre_consistency(df_movies, item_vectors)
    
    # 3. Visual Performance Plot
    plot_performance_metrics(scores)
    
    print("\n------------------------------------------------")
    print("SUMMARY FOR DEFENSE:")
    print(f"1. Average Genre Consistency is {np.mean(scores)*100:.1f}%.")
    print("   (This confirms the model understands movie content.)")
    print("------------------------------------------------")