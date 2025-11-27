# from flask import Flask, render_template, request
# import pickle
# import pandas as pd
# import requests

# app = Flask(__name__)

# # --- CONFIGURATION ---
# # Your TMDB API Key
# API_KEY = "dd9d0c38bee5a657640627c4d9265631"

# # --- LOAD FILES ---
# print("Loading model files...")
# try:
#     movies = pickle.load(open('movies.pkl', 'rb'))
#     similarity = pickle.load(open('similarity.pkl', 'rb'))
#     print("Files loaded successfully!")
# except FileNotFoundError:
#     print("❌ Error: .pkl files not found. Please run 'model_upgrade.py' first.")
#     # Initialize empty to prevent immediate crash if files are missing
#     movies = pd.DataFrame(columns=['title', 'movie_id'])
#     similarity = []

# def fetch_poster(movie_id):
#     """
#     Fetches the poster URL from TMDB API.
#     """
#     try:
#         url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
#         response = requests.get(url, timeout=5)
#         data = response.json()
#         if data.get('poster_path'):
#             return "https://image.tmdb.org/t/p/w500/" + data['poster_path']
#         else:
#             # Fallback image if poster is missing
#             return "https://via.placeholder.com/500x750?text=No+Image"
#     except Exception as e:
#         print(f"Error fetching poster: {e}")
#         return "https://via.placeholder.com/500x750?text=Error"

# def get_recommendations(movie_title):
#     # Create a mapping of lower-case titles to index
#     # ensuring all titles are strings to avoid errors
#     indices = pd.Series(movies.index, index=movies['title'].apply(lambda x: str(x).lower()))
    
#     movie_title = movie_title.lower().strip()
    
#     if movie_title not in indices:
#         return None
    
#     idx = indices[movie_title]
    
#     # --- CRITICAL FIX FOR "VALUE ERROR" ---
#     # If a movie exists multiple times in the dataset (e.g., "Batman"), 
#     # pandas returns a Series of indices instead of a single integer.
#     # We check for this and just take the first instance.
#     if isinstance(idx, pd.Series):
#         idx = idx.iloc[0]

#     # Get similarity scores for that specific movie index
#     sim_scores = list(enumerate(similarity[idx]))

#     # Sort them based on the similarity score (highest first)
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

#     # Get top 5 (skip index 0 because that is the movie itself)
#     sim_scores = sim_scores[1:6]

#     # Fetch Movie Details and Posters
#     recommended_movies = []
#     for i in sim_scores:
#         movie_id = movies.iloc[i[0]].movie_id
#         title = movies.iloc[i[0]].title
        
#         # Fetch the poster using the ID
#         poster_url = fetch_poster(movie_id)
        
#         recommended_movies.append({
#             'title': title, 
#             'id': movie_id,
#             'poster': poster_url
#         })
        
#     return recommended_movies

# @app.route('/')
# def index():
#     # Send the list of movies to the dropdown
#     return render_template('index.html', movie_list=movies['title'].values)

# @app.route('/recommend', methods=['POST'])
# def recommend():
#     movie_name = request.form.get('movie_name')
    
#     if not movie_name:
#         return render_template('index.html', movie_list=movies['title'].values, error="Please select a movie.")

#     recommendations = get_recommendations(movie_name)
    
#     if recommendations is None:
#         return render_template('index.html', movie_list=movies['title'].values, error=f"Movie '{movie_name}' not found in database!")
    
#     # Render the page with the recommendations
#     return render_template('index.html', 
#                          movie_list=movies['title'].values, 
#                          recommendations=recommendations, 
#                          movie_name=movie_name)

# if __name__ == '__main__':
#     app.run(debug=True)




from flask import Flask, render_template, request
import pickle
import pandas as pd
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# --- CONFIGURATION ---
# Your TMDB API Key
API_KEY = "dd9d0c38bee5a657640627c4d9265631"

# --- LOAD FILES ---
print("Loading model files...")
try:
    # 1. Load the Movie Titles/IDs
    movies = pickle.load(open('movies.pkl', 'rb'))
    
    # 2. Load the LightGCN Vectors (Embeddings)
    # Replaces the old 'similarity.pkl' which was too heavy/static
    item_embeddings = pickle.load(open('item_embeddings.pkl', 'rb'))
    
    print("Files loaded successfully!")
except FileNotFoundError:
    print("❌ Error: .pkl files not found. Please run 'model_upgrade.py' first.")
    # Initialize empty to prevent immediate crash
    movies = pd.DataFrame(columns=['title', 'movie_id'])
    item_embeddings = None

def fetch_poster(movie_id):
    """
    Fetches the poster URL from TMDB API.
    """
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
        response = requests.get(url, timeout=5)
        data = response.json()
        if data.get('poster_path'):
            return "https://image.tmdb.org/t/p/w500/" + data['poster_path']
        else:
            return "https://via.placeholder.com/500x750?text=No+Image"
    except Exception as e:
        print(f"Error fetching poster: {e}")
        return "https://via.placeholder.com/500x750?text=Error"

def get_recommendations(movie_title):
    # Create a mapping of lower-case titles to index
    # ensuring all titles are strings to avoid errors
    indices = pd.Series(movies.index, index=movies['title'].apply(lambda x: str(x).lower()))
    
    movie_title = movie_title.lower().strip()
    
    if movie_title not in indices:
        return None
    
    idx = indices[movie_title]
    
    # --- CRITICAL FIX FOR DUPLICATES ---
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]

    # --- NEW LIGHTGCN LOGIC ---
    # 1. Get the vector for the selected movie (Shape: 1 x 64)
    target_vector = item_embeddings[idx].reshape(1, -1)

    # 2. Calculate Similarity against ALL movies instantly
    # This replaces looking up a pre-calculated table
    sim_scores = cosine_similarity(target_vector, item_embeddings)[0]

    # 3. Create a list of (index, score) tuples
    sim_scores_list = list(enumerate(sim_scores))

    # 4. Sort (Highest similarity first)
    sim_scores_list = sorted(sim_scores_list, key=lambda x: x[1], reverse=True)

    # 5. Get top 5 (skip index 0 because that is the movie itself)
    top_matches = sim_scores_list[1:6]

    # Fetch Movie Details and Posters
    recommended_movies = []
    for i in top_matches:
        movie_idx = i[0]
        movie_id = movies.iloc[movie_idx].movie_id
        title = movies.iloc[movie_idx].title
        
        # Fetch the poster using the ID
        poster_url = fetch_poster(movie_id)
        
        recommended_movies.append({
            'title': title, 
            'id': movie_id,
            'poster': poster_url
        })
        
    return recommended_movies

@app.route('/')
def index():
    # Send the list of movies to the dropdown
    return render_template('index.html', movie_list=movies['title'].values)

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_name = request.form.get('movie_name')
    
    if not movie_name:
        return render_template('index.html', movie_list=movies['title'].values, error="Please select a movie.")

    recommendations = get_recommendations(movie_name)
    
    if recommendations is None:
        return render_template('index.html', movie_list=movies['title'].values, error=f"Movie '{movie_name}' not found in database!")
    
    # Render the page with the recommendations
    return render_template('index.html', 
                           movie_list=movies['title'].values, 
                           recommendations=recommendations, 
                           movie_name=movie_name)

if __name__ == '__main__':
    app.run(debug=True)