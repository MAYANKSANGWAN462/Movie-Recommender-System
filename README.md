# Movie Recommender System

This is a Flask-based Movie Recommender Web App that suggests similar movies to the one selected by the user. It uses a content-based recommendation model built with Python and TMDB API for fetching movie posters.

## 🧠 Features

✅ Interactive website with dropdown to select a movie  
✅ Fetches real posters and movie info from TMDB API  
✅ Shows top 5 similar movie recommendations  
✅ Simple Flask backend with Python and HTML frontend  

## 📁 Project Structure

```
Movie_Recommender_System/
├── app.py
├── movies.pkl
├── similarity.pkl
├── static/
│   └── style.css
└── templates/
    └── index.html
```

## ⚙️ Prerequisites

Before running this project, make sure you have:

- Python 3.10 or later installed
- Pip (Python package manager) installed
- Internet connection (to fetch posters from TMDB)

## 🚀 Setting Up the Project

### Step 1: Clone or Download the Project

If you downloaded the ZIP, extract it to a folder (e.g., `C:\Users\Desktop\Movie_Recommender_System`).

Or clone via Git:

```bash
git clone https://github.com/your-username/Movie_Recommender_System.git
cd Movie_Recommender_System
```

### Step 2: Create and Activate a Virtual Environment

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

Run this command to install everything required for the project:

```bash
pip install flask numpy pandas scikit-learn requests urllib3 certifi
```

(If you want to freeze them later)

```bash
pip freeze > requirements.txt
```

### Step 4: Verify Installation

To check if Flask is installed:

```bash
pip show flask
```

If it shows version info — you're good to go ✅

## 🔑 TMDB API Setup

This app uses the TMDB (The Movie Database) API to fetch posters.

1. Go to [https://www.themoviedb.org/](https://www.themoviedb.org/)
2. Create a free account
3. Go to Settings → API → Request an API Key
4. Fill in the form (use `http://localhost:5000` as your website URL)
5. Copy your API key (v3 auth)
6. Then open `app.py` and paste your API key here:

```python
API_KEY = "YOUR_TMDB_API_KEY"
```

## 🧩 Running the Project

### Step 1: Start the Flask App

Run:

```bash
python app.py
```

You should see:

```
* Running on http://127.0.0.1:5000
```

### Step 2: Open the App

Go to your browser and open:

```
http://127.0.0.1:5000
```

### Step 3: Use the Web App

1. Select a movie name from the dropdown
2. Click "Recommend"
3. The app will show 5 similar movies with posters 🎥

## 💡 Common Issues and Fixes

| Issue | Solution |
|-------|----------|
| ModuleNotFoundError: No module named 'flask' | Run `pip install flask` |
| SSL Error (SSLError) | Run `pip install --upgrade requests urllib3 certifi` |
| No filter named 'zip' | Use the updated index.html provided in this repo |
| Posters not loading | Check your TMDB API key in app.py |
| App not running | Ensure venv is activated (`venv\Scripts\activate`) |

## 🎨 Optional: Style (CSS)

Create `static/style.css` (optional):

```css
body {
  font-family: Arial, sans-serif;
  background-color: #121212;
  color: white;
  text-align: center;
  padding: 30px;
}

select, button {
  padding: 10px;
  font-size: 16px;
  margin-top: 10px;
  border-radius: 8px;
}

.movie-grid {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 20px;
  margin-top: 20px;
}

.movie-item img {
  border-radius: 10px;
  transition: transform 0.3s;
}

.movie-item img:hover {
  transform: scale(1.05);
}
```

## 🧰 Requirements Summary

If you want to include a `requirements.txt`, here's what to put inside:

```
flask
numpy
pandas
scikit-learn
requests
urllib3
certifi
```

Then others can simply do:

```bash
pip install -r requirements.txt
```

## ✅ Stop the App

To stop the server, press **CTRL + C** in the terminal.

## 💻 Deploying (Optional)

Later, you can deploy this app online using:

- Render
- PythonAnywhere
- Vercel

## 👨‍💻 Author

**Developer:** Your Name  
**Email:** your.email@example.com  

**Description:** Flask-based Movie Recommendation System using TMDB API.
