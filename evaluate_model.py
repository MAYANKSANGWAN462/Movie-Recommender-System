import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate():
    print("Loading model files...")
    try:
        movies = pickle.load(open('movies.pkl', 'rb'))
        similarity = pickle.load(open('similarity.pkl', 'rb'))
    except FileNotFoundError:
        print("Error: .pkl files not found. Run model_upgrade.py first!")
        return

    print(f"Loaded {len(movies)} movies.")

    # --- METRIC: GENRE CONSISTENCY CHECK ---
    print("Checking consistency on 100 random movies...")
    sample_indices = np.random.choice(movies.index, size=100, replace=False)
    precision_scores = []

    for idx in sample_indices:
        # Get top 5 similar movies (excluding self)
        scores = list(enumerate(similarity[idx]))
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]
        
        source_genres = set(movies.iloc[idx]['genres'])
        hits = 0
        
        for rec_idx, score in sorted_scores:
            rec_genres = set(movies.iloc[rec_idx]['genres'])
            # If genres overlap (e.g. Action matches Action), it's a hit
            if not source_genres.isdisjoint(rec_genres):
                hits += 1
        
        precision_scores.append(hits / 5.0)

    avg_precision = np.mean(precision_scores)
    print(f"\n------------------------------------------------")
    print(f"MODEL ACCURACY (Genre Match): {avg_precision * 100:.2f}%")
    print(f"------------------------------------------------")

    # --- VISUALIZATION ---
    print("Generating graphs...")
    plt.figure(figsize=(12, 5))

    # Graph 1: Similarity Score Distribution
    # Helps you see if the model is too strict or too loose
    plt.subplot(1, 2, 1)
    # Sample 10,000 scores for speed
    flat_scores = similarity.flatten()
    sample_scores = np.random.choice(flat_scores, size=10000, replace=False)
    sns.histplot(sample_scores, bins=50, kde=True, color='#4c72b0')
    plt.title('Distribution of Similarity Scores')
    plt.xlabel('Similarity Score (0=Different, 1=Same)')

    # Graph 2: Accuracy Distribution
    plt.subplot(1, 2, 2)
    sns.histplot(precision_scores, bins=5, color='#55a868', kde=False)
    plt.title('Recommendation Quality (Right is Better)')
    plt.xlabel('Precision Score (0.0 to 1.0)')
    plt.ylabel('Count of Movies')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate()