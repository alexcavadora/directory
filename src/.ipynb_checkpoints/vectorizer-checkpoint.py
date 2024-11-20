import math
from collections import Counter, defaultdict
import numpy as np
import spacy

class Vectorizer:
    def __init__(self):
        self.idf = {}
        self.vocabulary = {}
        self.nlp = spacy.load("es_core_news_sm")  # Load the Spanish language model

    def preprocess(self, description):
        # Use SpaCy for tokenization, lemmatization, and stop word removal
        doc = self.nlp(description)
        tokens = [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]
        return tokens

    def fit_transform(self, descriptions):
        # Preprocess descriptions
        descriptions = [' '.join(self.preprocess(desc)) for desc in descriptions]
        #print("Preprocessed Descriptions:", descriptions)  # Debugging line

        # Calculate TF for each document
        tf_documents = [Counter(doc.split()) for doc in descriptions]

        # Calculate document frequency (DF) for IDF
        doc_count = len(descriptions)
        df = defaultdict(int)
        for tf in tf_documents:
            for word in tf:
                df[word] += 1

        # Calculate IDF
        self.idf = {word: math.log(doc_count / (1 + freq)) for word, freq in df.items()}
        #print("IDF Values:", self.idf)  # Debugging line

        # Create vocabulary
        self.vocabulary = {word: idx for idx, word in enumerate(self.idf.keys())}

        # Calculate TF-IDF matrix
        tf_idf_matrix = []
        for tf in tf_documents:
            row = [tf[word] * self.idf[word] if word in tf else 0 for word in self.vocabulary]
            tf_idf_matrix.append(row)

        tf_idf_matrix = np.array(tf_idf_matrix)
        print("TF-IDF Matrix Shape:", tf_idf_matrix.shape)

        return tf_idf_matrix

    def transform(self, query):
        # Preprocess the query to tokenize, lemmatize, and remove stop words
        preprocessed_query = self.preprocess(query)

        # Join the preprocessed tokens back into a single string
        query = ' '.join(preprocessed_query)

        # Calculate term frequency (TF) for the preprocessed query
        query_tf = Counter(query.split())

        # Create the TF-IDF vector for the query
        tf_idf_vector = [query_tf[word] * self.idf.get(word, 0) for word in self.vocabulary]

        return np.array(tf_idf_vector)
