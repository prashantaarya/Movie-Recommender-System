import streamlit as st
import pickle
import requests
from movie_recommender import recommend  # Import the recommend function

# Cache the model loading function to avoid reloading on each interaction
@st.cache_data  # Remove allow_output_mutation argument
def load_model():
    movies = pickle.load(open(r'C:\Users\RYR3COB\Desktop\Movie-Recommender-System\model\movie_list.pkl', 'rb'))  # Load movies list
    similarity = pickle.load(open(r'C:\Users\RYR3COB\Desktop\Movie-Recommender-System\model\similarity.pkl', 'rb'))  # Load precomputed similarity matrix
    return movies, similarity

# Function to fetch the movie poster using TMDB API
@st.cache_resource  # Use st.cache_resource for caching external resources like API calls
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = f"https://image.tmdb.org/t/p/w500/{poster_path}"
    return full_path

# Function to fetch IMDb URL from TMDB API (or stored movie data)
@st.cache_resource
def fetch_imdb_url(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
    data = requests.get(url)
    data = data.json()
    imdb_id = data['imdb_id']  # TMDB provides the IMDb ID
    imdb_url = f"https://www.imdb.com/title/{imdb_id}/" if imdb_id else ""
    return imdb_url

# Load the necessary data and models only once
movies, similarity = load_model()

# Streamlit Interface
st.header('Movie Recommender System')

# Movie list for dropdown selection
movie_list = movies['title'].values
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)

if st.button('Show Recommendation'):
    recommended_movie_names = recommend(selected_movie, top_n=5)
    col1, col2, col3, col4, col5 = st.columns(5)

    # For each recommended movie, display the movie name, poster, and clickable IMDb link
    for i, col in enumerate([col1, col2, col3, col4, col5]):
        movie_name = recommended_movie_names[i]
        movie_id = movies[movies['title'] == movie_name].movie_id.values[0]
        poster_url = fetch_poster(movie_id)
        imdb_url = fetch_imdb_url(movie_id)

        with col:
            st.text(movie_name)
            st.image(poster_url)
            if imdb_url:
                st.markdown(f"[IMDb Link]({imdb_url})", unsafe_allow_html=True)
            else:
                st.text("IMDb Link not available.")
