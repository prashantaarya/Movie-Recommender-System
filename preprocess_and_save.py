from movie_recommender import movies, hybrid_similarity
from data_processing import save_model

# Save the movies DataFrame and similarity matrix in the current directory
save_model(movies, r'C:\Users\RYR3COB\Desktop\Movie-Recommender-System\model\movie_list.pkl')
save_model(hybrid_similarity, r'C:\Users\RYR3COB\Desktop\Movie-Recommender-System\model\similarity.pkl')
