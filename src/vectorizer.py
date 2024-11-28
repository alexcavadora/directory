import math
from collections import Counter, defaultdict
import numpy as np
import spacy
import pandas as pd
class Vectorizer:
    def __init__(self):
        self.idf = {}
        self.vocabulary = {}
        self.nlp = spacy.load("es_core_news_lg")
        self.sentences_split = []

    def preprocess(self, text):
        # Tokenization, lemmatization, and stop word removal
        doc = self.nlp(text)
        tokens = [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]
        return tokens

    def split_sentences(self, document):
        doc = self.nlp(document)
        sentences = [sent.text for sent in doc.sents]
        
        return sentences

    def fit_transform(self, documents):
        # Split documents into sentences and preprocess
        all_sentences = [sent for doc in documents for sent in self.split_sentences(doc)]
        
        preprocessed_sentences = [' '.join(self.preprocess(sent)) for sent in all_sentences]
        self.sentences_split = preprocessed_sentences
        # Compute TF-IDF
        tf_documents = [Counter(sent.split()) for sent in preprocessed_sentences]
        doc_count = len(preprocessed_sentences)
        df = defaultdict(int)
        for tf in tf_documents:
            for word in tf:
                df[word] += 1

        self.idf = {word: math.log(doc_count / (1 + freq)) for word, freq in df.items()}
        self.vocabulary = {word: idx for idx, word in enumerate(self.idf.keys())}
        #pd.DataFrame(list(self.vocabulary.keys()), columns=['word']).to_csv('vocabulary.csv', index=False)
        tf_idf_matrix = []
        for tf in tf_documents:
            row = [tf[word] * self.idf[word] if word in tf else 0 for word in self.vocabulary]
            tf_idf_matrix.append(row)

        return np.array(tf_idf_matrix), all_sentences

    def transform_sentence(self, sentence):
        preprocessed = self.preprocess(sentence)
        query = ' '.join(preprocessed)
        query_tf = Counter(query.split())
        tf_idf_vector = [query_tf[word] * self.idf.get(word, 0) for word in self.vocabulary]
        return np.array(tf_idf_vector)
