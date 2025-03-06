# AI-Insights
Simple python snippets to experience how LLMs could assist visualise qualitative data.

## similarity_test_ollama.py
Computes cosine similarity between sentence pairs.

## vis_embedgs_ollama.py
Visualise text (customer feedback) embeddings using 2 dimensional Principal Component Analysis (PCA), a dimensionality reduction technique.

**Steps:**
1. Reads a CSV file (sentiment-analysis.csv) taken from Kaggel containing customer feedback texts.
2. Extract the required text as input to the LLM.
3. Generate embeddings using 'all-MiniLM-L6-v2' model.
4. Reduce dimensions to 2D for visualisation.
5. Plot results.
