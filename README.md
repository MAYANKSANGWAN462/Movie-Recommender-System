
# 🎬 Hybrid Movie Recommender System

A robust Movie Recommendation Engine utilizing a **Hybrid Approach** (Content-Based Filtering + BERT Embeddings). It combines **TF-IDF** for keyword matching and **Sentence-Transformers** for semantic understanding to suggest movies based on metadata and plot summaries.

## 🧠 Key Features

✅ **Hybrid Engine:** Combines TF-IDF (Genres/Cast) + BERT (Plot Semantics) + Popularity scores.
✅ **Model Evaluation:** Includes a dedicated script to visualize model accuracy and distribution.
✅ **Interactive Web App:** Fetches high-quality posters via the TMDB API.
✅ **Modular Architecture:** Separates Model Training (`model_upgrade.py`) from the Web App (`app.py`).

## 📁 Project Structure

```text
Movie_Recommender_System/
├── model_upgrade.py     
├── evaluate_model.py     
├── app.py                
├── tmdb_5000_movies.csv  
├── tmdb_5000_credits.csv 
├── movies.pkl           
├── similarity.pkl        
├── requirements.txt      
├── static/
│   └── style.css
└── templates/
    └── index.html
````

## ⚙️ Prerequisites

  * Python 3.10 or later
  * Pip (Python package manager)
  * An active internet connection (to fetch posters and download BERT models)

## 🚀 Setting Up the Project

### Step 1: Clone the Repository

```bash
git clone [https://github.com/your-username/Movie_Recommender_System.git](https://github.com/your-username/Movie_Recommender_System.git)
cd Movie_Recommender_System
```

### Step 2: Create Virtual Environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install All Dependencies

Run this **single command** to install the web framework, machine learning tools, plotting libraries, and security certificates:

```bash
pip install flask pandas numpy scikit-learn sentence-transformers matplotlib seaborn requests urllib3 certifi
```

### Step 4: TMDB API Setup

1.  Go to [TheMovieDB.org](https://www.themoviedb.org/) and sign up.
2.  Request an API Key (Settings → API).
3.  Open `app.py` and paste your key:
    ```python
    API_KEY = "YOUR_TMDB_API_KEY_HERE"
    ```

-----

## 🛠️ Workflow: How to Run

Follow this order to build, test, and run the system.

### 1\. Train the Model (The Factory)

Runs the hybrid algorithm, downloads BERT, and creates the `.pkl` files.

```bash
python model_upgrade.py
```

*Output: Generates `movies.pkl` and `similarity.pkl`*

### 2\. Evaluate Performance (The QA)

Checks genre consistency and generates accuracy graphs (`model_evaluation_report.png`).

```bash
python evaluate_model.py
```

*Output: Prints Accuracy % and shows visual graphs.*

### 3\. Launch the App (The Shop)

Starts the Flask web server.

```bash
python app.py
```

*Access the app at:* `http://127.0.0.1:5000`

-----

## 💡 Common Issues and Fixes

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'matplotlib'` | Run the pip install command in Step 3 again. |
| `IndexError: index out of bounds` | Ensure `model_upgrade.py` includes `reset_index(drop=True)` after dropping NA values. |
| `SSLError` or Connection Error | `urllib3` and `certifi` are included in Step 3 to fix this. |
| Posters loading inconsistently | Check `app.py` for connection timeouts or invalid API Key. |
| System freezes during training | BERT is heavy. Ensure you have 8GB+ RAM. |

## 🧰 Requirements Summary

If you prefer using a `requirements.txt` file, copy this content inside it:

```text
flask
pandas
numpy
scikit-learn
sentence-transformers
matplotlib
seaborn
requests
urllib3
certifi
```

Then run: `pip install -r requirements.txt`

-----

## 👨‍💻 Author

**Developer:** Mayank Sangwan & Sameer Verma
**Email:** sangwanmayank462@gmail.com

