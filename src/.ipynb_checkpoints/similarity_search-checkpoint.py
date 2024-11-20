import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class SimilaritySearch:
    def __init__(self, names, descriptions, description_vectors, vectorizer):
        self.names = names
        self.descriptions = descriptions
        self.description_vectors = description_vectors
        self.vectorizer = vectorizer

    def find_similar(self, query, top_n=20):
        # Preprocess query
        query_vector = self.vectorizer.transform(query)

        # Compute similarities for the query vector against all description vectors
        similarities = cosine_similarity(query_vector.reshape(1, -1), self.description_vectors).flatten()

        top_indices = np.argsort(similarities)[-top_n:][::-1]
        non_zero_indices = [i for i in top_indices if similarities[i] > 0]

        return pd.DataFrame({
            'Names': [self.names[i] for i in non_zero_indices],
            'Description': [self.descriptions[i] for i in non_zero_indices],
            'Length': [len(self.descriptions[i]) for i in non_zero_indices],
            'Similarity': [similarities[i] for i in non_zero_indices]
        })
