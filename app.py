import streamlit as st
import pandas as pd
import requests
import time

TMDB_API_KEY = st.secrets["TMDB_API_KEY"]

@st.cache_data
def load_data():
    sim_df = pd.read_pickle("collab_model.pkl")
    movies_df = pd.read_pickle("movies_list.pkl")
    return sim_df, movies_df

def fetch_poster(tmdb_id):
    if pd.isna(tmdb_id):
        return None, "Missing ID in CSV file"
        
    try:
        url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={TMDB_API_KEY}&language=en-US"
        response = requests.get(url, timeout=5)
        
        # If TMDB blocks us, grab their exact error message
        if response.status_code != 200:
            return None, f"TMDB Blocked Us (Error {response.status_code})"
            
        data = response.json()
        if data.get('poster_path'):
            return "https://image.tmdb.org/t/p/w500" + data['poster_path'], "Success"
            
        return None, "TMDB has no poster for this movie"
        
    except Exception as e:
        return None, f"Code crashed: {str(e)}"

sim_df, movies_df = load_data()

st.title("🎥 AI Movie Recommender (Diagnostic Mode)")

valid_movies = movies_df[movies_df['movie_id'].isin(sim_df.index)]
movie_list = sorted(valid_movies['title'].dropna().unique().tolist())
selected_movie = st.selectbox("Search for a movie you watched:", movie_list)

if st.button("Get Recommendations"):
    seed_row = movies_df[movies_df['title'] == selected_movie].iloc[0]
    seed_id = seed_row['movie_id']
    
    if seed_id in sim_df.index:
        st.subheader(f"Because you watched {selected_movie}:")
        
        sim_scores = sim_df[seed_id].drop(labels=[seed_id], errors='ignore')
        top_15_ids = sim_scores.nlargest(15).index.tolist()
        
        valid_recommendations = []
        
        for m_id in top_15_ids:
            movie_data = movies_df[movies_df['movie_id'] == m_id].iloc[0]
            title = movie_data['title']
            tmdb_id = movie_data['tmdbId']
            
            # Unpack the URL and the Error Message
            poster_url, debug_msg = fetch_poster(tmdb_id)
            time.sleep(0.3)
            
            if poster_url is not None:
                valid_recommendations.append((title, poster_url))
            else:
                # PRINT THE ERROR DIRECTLY TO THE WEBSITE SCREEN
                st.error(f"Failed to load poster for '{title}': {debug_msg}")
                
            if len(valid_recommendations) == 5:
                break
        
        cols = st.columns(5)
        for i, (title, poster_url) in enumerate(valid_recommendations):
            with cols[i]:
                st.image(poster_url)
                st.caption(title)
