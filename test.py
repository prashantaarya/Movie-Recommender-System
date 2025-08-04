from sentence_transformers import SentenceTransformer

# This will download the model and cache it in ~/.cache/torch/sentence_transformers
model = SentenceTransformer('all-MiniLM-L6-v2')
# Path to the cached model directory
sbert_model = SentenceTransformer('./path/to/all-MiniLM-L6-v2')

