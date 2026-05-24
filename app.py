import streamlit as st
import pandas as pd
import requests
import math

# --- ADD YOUR TMDB API KEY HERE ---
TMDB_API_KEY = "f2bb976d39b05aa2170dafde25d9a700"

@st.cache_data
def load_data():
    sim_df = pd.read_pickle("collab_model.pkl")
    movies_df = pd.read_pickle("movies_list.pkl")
    return sim_df, movies_df

def fetch_poster(tmdb_id):
    fallback_url = "https://upload.wikimedia.org/wikipedia/commons/6/65/No-Image-Placeholder.svg"
    if pd.isna(tmdb_id):
        return fallback_url
    try:
        url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={TMDB_API_KEY}&language=en-US"
        response = requests.get(url)
        data = response.json()
        if 'poster_path' in data and data['poster_path']:
            return "https://image.tmdb.org/t/p/w500" + data['poster_path']
        else:
            return fallback_url
    except Exception:
        return fallback_url

sim_df, movies_df = load_data()

st.title("🎥 AI Movie Recommender")
st.markdown("Powered by Custom Collaborative Filtering")

# --- THE FIX IS HERE ---
# Filter the dropdown to ONLY show movies that exist inside our trained sim_df matrix
valid_movies = movies_df[movies_df['movie_id'].isin(sim_df.index)]
movie_list = sorted(valid_movies['title'].dropna().unique().tolist())

selected_movie = st.selectbox("Search for a movie you watched:", movie_list)

if st.button("Get Recommendations"):
    seed_row = movies_df[movies_df['title'] == selected_movie].iloc[0]
    seed_id = seed_row['movie_id']
    
    if seed_id in sim_df.index:
        st.subheader(f"Because you watched {selected_movie}:")
        sim_scores = sim_df[seed_id].drop(labels=[seed_id], errors='ignore')
        top_5_ids = sim_scores.nlargest(5).index.tolist()
        
        cols = st.columns(5)
        for i, m_id in enumerate(top_5_ids):
            movie_data = movies_df[movies_df['movie_id'] == m_id].iloc[0]
            title = movie_data['title']
            tmdb_id = movie_data['tmdbId']
            poster_url = fetch_poster(tmdb_id)
            
            with cols[i]:
                st.image(poster_url)
                st.caption(title)
    else:
        st.warning("Not enough user data to make a recommendation.")
