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
        return None

    try:
        url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={TMDB_API_KEY}&language=en-US"
        response = requests.get(url, timeout=5)
        data = response.json()

        if data.get('poster_path'):
            return "https://image.tmdb.org/t/p/w500" + data['poster_path']
        return None
    except Exception:
        return None

sim_df, movies_df = load_data()

st.title("🎥 AI Movie Recommender")
st.markdown("Powered by Custom Item-Based Collaborative Filtering")

# --- NEW UI HOVER EFFECT CODE ---
st.markdown('''
    <style>
        /* Smooth transition for the zoom */
        div[data-testid="stImage"] img {
            transition: transform 0.3s ease-in-out;
            border-radius: 8px; /* Rounds the corners slightly */
        }
        /* What happens when the mouse hovers over it */
        div[data-testid="stImage"] img:hover {
            transform: scale(1.08); /* Zooms in 8% */
            box-shadow: 0px 10px 20px rgba(0,0,0,0.6); /* Adds a dark shadow */
        }
    </style>
''', unsafe_allow_html=True)
# --------------------------------

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

            poster_url = fetch_poster(tmdb_id)
            time.sleep(0.3)

            if poster_url is not None:
                valid_recommendations.append((title, poster_url))

            if len(valid_recommendations) == 5:
                break

        cols = st.columns(5)
        for i, (title, poster_url) in enumerate(valid_recommendations):
            with cols[i]:
                st.image(poster_url)
                st.caption(title)
    else:
        st.warning("Not enough user data to make a recommendation.")
