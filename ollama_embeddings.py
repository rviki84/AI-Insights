from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

text = "Ollama is great for local AI models!"
embedding = model.encode(text)

print("Embedding shape:", embedding.shape)  # Should be (384,) for this model
print("First 5 values:", embedding[:5])  # Print first 5 values
