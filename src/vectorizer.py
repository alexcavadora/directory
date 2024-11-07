import math
from collections import Counter, defaultdict
import numpy as np
class Vectorizer:
    def __init__(self):
        self.idf = {}
        self.vocabulary = {}

    def fit_transform(self, descriptions):
        # Tokenize and lower each description
        descriptions = [desc.lower().split() for desc in descriptions]

        # Calculate TF for each document
        tf_documents = [Counter(doc) for doc in descriptions]

        # Calculate document frequency (DF) for IDF
        doc_count = len(descriptions)
        df = defaultdict(int)
        for tf in tf_documents:
            for word in tf:
                df[word] += 1

        # Calculate IDF
        self.idf = {word: math.log(doc_count / (1 + freq)) for word, freq in df.items()}
        self.vocabulary = {word: idx for idx, word in enumerate(self.idf.keys())}

        # Calculate TF-IDF matrix
        tf_idf_matrix = []
        for tf in tf_documents:
            row = [tf[word] * self.idf[word] if word in tf else 0 for word in self.vocabulary]
            tf_idf_matrix.append(row)

        return np.array(tf_idf_matrix)

    def transform(self, query):
        # Tokenize and lower the query
        query = query.lower().split()
        query_tf = Counter(query)

        # Convert query to TF-IDF vector
        tf_idf_vector = [query_tf[word] * self.idf.get(word, 0) for word in self.vocabulary]
        return np.array(tf_idf_vector)
