import numpy as np
import pandas as pd

class SimilaritySearch:
    def __init__(self, names, descriptions, description_vectors):
        self.names = names
        self.descriptions = descriptions
        self.description_vectors = description_vectors

    def cosine_similarity(self, vec1, vec2):
        # Calculate cosine similarity between two vectors
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2) if norm_vec1 and norm_vec2 else 0

    def find_similar(self, query_vector, top_n=20):
        # Compute similarities for the query vector against all description vectors
        similarities = [self.cosine_similarity(query_vector, doc_vec) for doc_vec in self.description_vectors]

        top_indices = np.argsort(similarities)[-top_n:][::-1]

        non_zero_indices = [i for i in top_indices if similarities[i] > 0]

        return pd.DataFrame({
            'Names' : [self.names[i] for i in non_zero_indices],
            'Description': [self.descriptions[i] for i in non_zero_indices],
            'Similarity':  [similarities[i] for i in non_zero_indices]
            })
