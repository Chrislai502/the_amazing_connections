from sentence_transformers import SentenceTransformer
import numpy as np
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(sentences)
embedding1, embedding2 = embeddings[0], embeddings[1]

# Compute cosine similarity
# Note: The embeddings are already normalized, so we can just take the dot product
similarity = np.dot(embedding1, embedding2)

# Normalize from [-1, 1] to [0, 1] range
normalized_similarity = (similarity + 1) / 2

print(float(normalized_similarity))

