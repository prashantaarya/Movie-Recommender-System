import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import ast

# Load the movies data
movies = pd.read_csv(r'C:\Users\RYR3COB\Desktop\Movie-Recommender-System\data\tmdb_5000_movies.csv')
credits = pd.read_csv(r'C:\Users\RYR3COB\Desktop\Movie-Recommender-System\data\tmdb_5000_credits.csv')

# Merge movies with credits data
movies = movies.merge(credits, on='title')

# Preprocess the data
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

# Create a function to handle the cast
def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

movies['cast'] = movies['cast'].apply(convert3)

# Create function to extract directors
def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

movies['crew'] = movies['crew'].apply(fetch_director)

# Combine features into a tag list for each movie, ensuring we handle them as strings
movies['tags'] = movies['overview'].fillna('') + \
                 movies['genres'].apply(lambda x: " ".join(x)).fillna('') + \
                 movies['keywords'].apply(lambda x: " ".join(x)).fillna('') + \
                 movies['cast'].apply(lambda x: " ".join(x)).fillna('') + \
                 movies['crew'].apply(lambda x: " ".join(x)).fillna('')

new_df = movies[['movie_id', 'title', 'tags']]

# Convert the tags column to lowercase
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

# Load SBERT model
sbert_model = SentenceTransformer('./hybrid_model')  # Load from the saved model

#sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
#sbert_model.save('./hybrid_model')

# Compute SBERT embeddings for 'tags' column
sbert_embeddings = sbert_model.encode(new_df['tags'].tolist())
sbert_embeddings = normalize(sbert_embeddings)

# Compute TF-IDF Similarity for 'tags' column
tfidf_vectorizer = CountVectorizer(max_features=5000, stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(new_df['tags'])
tfidf_similarity = cosine_similarity(tfidf_matrix)

# Compute Hybrid Similarity
alpha = 0.7  # Adjust weight (higher means more SBERT influence)
hybrid_similarity = alpha * sbert_embeddings @ sbert_embeddings.T + (1 - alpha) * tfidf_similarity

def recommend(movie, top_n=5):
    movie_index = new_df[new_df['title'] == movie].index[0]  # Find movie index by title
    distances = hybrid_similarity[movie_index]  # Get hybrid similarity scores
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:top_n+1]  # Get top N recommendations
    
    recommended_movie_names = [new_df.iloc[i[0]].title for i in movies_list]
    return recommended_movie_names
