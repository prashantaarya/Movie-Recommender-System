# data_processing.py

import pandas as pd
import pickle

def load_data():
    movies = pd.read_csv(r'C:\Users\RYR3COB\Desktop\Movie-Recommender-System\data\tmdb_5000_movies.csv')
    credits = pd.read_csv(r'C:\Users\RYR3COB\Desktop\Movie-Recommender-System\data\tmdb_5000_credits.csv')
    return movies, credits

def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)
