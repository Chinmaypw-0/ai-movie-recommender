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
    # Strict filter: If ID is missing, return None immediately
    if pd.isna(tmdb_id):
        return None
        
    try:
        url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={TMDB_API_KEY}&language=en-US"
        response = requests.get(url, timeout=5)
        data = response.json()
        
        # Strict filter: Ensure 'poster_path' actually exists and has data
        if data.get('poster_path'):
            return "https://image.tmdb.org/t/p/w500" + data['poster_path']
        
        return None 
        
    except Exception:
        return None 

sim_df, movies_df = load_data()

st.title("🎥 AI Movie Recommender")
st.markdown("Powered by Custom Item-Based Collaborative Filtering")

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
            time.sleep(0.3) # 300ms pause to protect against TMDB rate limits
            
            # The exact condition: Only keep the movie if a real poster URL was found
            if poster_url is not None:
                valid_recommendations.append((title, poster_url))
                
            # Stop as soon as we secure 5 good movies
            if len(valid_recommendations) == 5:
                break
        
        # Display the final guaranteed valid posters
        cols = st.columns(5)
        for i, (title, poster_url) in enumerate(valid_recommendations):
            with cols[i]:
                st.image(poster_url)
                st.caption(title)
    else:
        st.warning("Not enough user data to make a recommendation.")
