from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "Ollama is great for local AI models.",
    "Machine learning powers AI applications.",
    "The cat sat on the mat.",
    "Deep learning improves computer vision."
]

# Generate embeddings
embeddings = model.encode(sentences)

# Reduce dimensions to 2D for visualization
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

# Plot results
plt.figure(figsize=(6, 5))
for i, sentence in enumerate(sentences):
    plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1])
    plt.text(reduced_embeddings[i, 0], reduced_embeddings[i, 1], sentence[:10], fontsize=9)

plt.xlabel("PCA Dim 1")
plt.ylabel("PCA Dim 2")
plt.title("Sentence Embeddings Visualization")
plt.show()
