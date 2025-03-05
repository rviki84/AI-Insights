from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "AI models are becoming more powerful.",
    "Machine learning is advancing quickly.",
    "The weather is nice today."
]

# Generate embeddings
embeddings = model.encode(sentences)

# Compute cosine similarity between sentence pairs
similarity_matrix = cosine_similarity([embeddings[0]], embeddings[1:])

print("Similarity with 2nd sentence:", similarity_matrix[0][0])  # High (~0.8+)
print("Similarity with 3rd sentence:", similarity_matrix[0][1])  # Low (~0.2)
