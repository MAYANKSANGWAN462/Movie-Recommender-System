import pandas as pd
import numpy as np
import ast
import pickle
from sentence_transformers import SentenceTransformer

# ==========================================
# CONFIGURATION
# ==========================================
# We use 'all-mpnet-base-v2'. It is larger and more accurate than 'miniLM'.
# It maps movies to a 768-dimensional vector space based on content.
MODEL_NAME = 'all-mpnet-base-v2' 

def parse_list(obj):
    """Helper to safely convert stringified lists to Python lists."""
    try:
        if pd.isna(obj): return []
        return [i['name'] for i in ast.literal_eval(obj)]
    except (ValueError, SyntaxError):
        return []

def fetch_director(obj):
    """Extracts the director's name."""
    try:
        if pd.isna(obj): return []
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                return [i['name']]
    except (ValueError, SyntaxError):
        pass
    return []

def load_and_clean_data():
    print("--- Step 1: Loading TMDB Dataset ---")
    try:
        movies = pd.read_csv('tmdb_5000_movies.csv')
        credits = pd.read_csv('tmdb_5000_credits.csv')
    except FileNotFoundError:
        print("❌ Error: CSV files not found. Make sure 'tmdb_5000_movies.csv' and 'tmdb_5000_credits.csv' are in the folder.")
        exit()

    # Merge datasets on title
    df = movies.merge(credits, on='title')
    
    # Select only the columns we need for the engine
    df = df[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew', 'vote_count']]
    
    # Filter: Remove movies with very few votes (usually bad quality/noise)
    df = df[df['vote_count'] > 20].reset_index(drop=True)
    
    df.dropna(subset=['overview'], inplace=True)
    
    print(f"✅ Data Loaded: {len(df)} movies ready for processing.")
    return df

def create_rich_semantic_tags(df):
    print("--- Step 2: Feature Engineering (Creating 'Semantic Soup') ---")
    
    # 1. Parse JSON columns
    df['genres'] = df['genres'].apply(parse_list)
    df['keywords'] = df['keywords'].apply(parse_list)
    df['cast'] = df['cast'].apply(lambda x: parse_list(x)[:4]) # Increased to top 4 actors
    df['crew'] = df['crew'].apply(fetch_director)

    # 2. Advanced Text Processing
    # We collapse spaces (e.g., "Johnny Depp" -> "JohnnyDepp") 
    # This creates a unique "token" for the actor, distinct from "Johnny" (generic name)
    df['genres'] = df['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
    df['keywords'] = df['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
    df['cast'] = df['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
    df['crew'] = df['crew'].apply(lambda x: [i.replace(" ", "") for i in x])
    
    # 3. Construct the "Semantic Soup"
    # We weight the components by repeating important features.
    # Structure: [Title] + [Overview] + [Genre] + [Keywords] + [Cast] + [Director]
    def create_soup(x):
        return (
            x['title'] + " " +       # Title helps with sequels (Harry Potter 1 & 2)
            x['overview'] + " " + 
            " ".join(x['genres']) + " " + 
            " ".join(x['keywords']) + " " + 
            " ".join(x['cast']) + " " + 
            " ".join(x['crew'])
        )
    
    df['tags'] = df.apply(create_soup, axis=1)
    
    # Prepare export dataframe (Lite version for the App)
    df_export = df[['movie_id', 'title', 'genres']].reset_index(drop=True)
    
    return df_export, df['tags'].tolist()

def generate_embeddings(text_tags):
    print(f"--- Step 3: Generating Deep Learning Embeddings ({MODEL_NAME}) ---")
    print("    (This will download the model (~400MB) and process. It may take 1-2 minutes.)")
    
    # Initialize the Transformer
    model = SentenceTransformer(MODEL_NAME)
    
    # Inference: Convert text descriptions to Dense Vectors (Embeddings)
    embeddings = model.encode(text_tags, show_progress_bar=True)
    
    print(f"✅ Vectors Generated. Shape: {embeddings.shape}")
    return embeddings

if __name__ == "__main__":
    # Execute Pipeline
    df_raw = load_and_clean_data()
    df_export, text_tags = create_rich_semantic_tags(df_raw)
    item_vectors = generate_embeddings(text_tags)
    
    # Save Artifacts
    print("--- Step 4: Saving Artifacts ---")
    pickle.dump(df_export, open('movies.pkl', 'wb'))
    pickle.dump(item_vectors, open('item_embeddings.pkl', 'wb'))
    
    print("\n🎉 SUCCESS: Model Upgrade Complete.")
    print("   Files generated: 'movies.pkl', 'item_embeddings.pkl'")
    print("   You can now run 'app.py' or 'evaluate_model.py'.")