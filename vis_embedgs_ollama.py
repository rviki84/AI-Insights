from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import csv

print("Reading the CSV file containing customer feedback text...")

with open("sentiment-analysis.csv", mode="r", encoding="utf-8") as file:
    reader = csv.reader(file)
    first_column = [row[0] for row in reader]

model = SentenceTransformer('all-MiniLM-L6-v2')

'''
sentences = [
    "Ollama is great for local AI models.",
    "Machine learning powers AI applications.",
    "The cat sat on the mat.",
    "Deep learning improves computer vision."
]
'''

# Get the count of texts to generate embeddings
textCount = int(input("Enter count of feedback text to generate embeddings: "))
text = []  # List to store first values
rows = first_column[1:textCount+1]
for row in rows:
    textstrip = [item.strip() for item in row.split(",") if item]
    text.append(textstrip[0])

# Generate embeddings
embeddings = model.encode(text)

# Reduce dimensions to 2D for visualization
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

# Plot results
plt.figure(figsize=(12, 6))
for i, sentence in enumerate(text):
    plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1])
    plt.text(reduced_embeddings[i, 0], reduced_embeddings[i, 1], sentence[:50], fontsize=7)

plt.xlabel("PCA Dim 1")
plt.ylabel("PCA Dim 2")
plt.title("Text Embeddings Visualization")
plt.show()
