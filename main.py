import sys
from src.similarity_search import SimilaritySearch
from src.vectorizer import Vectorizer
import pandas as pd

if len(sys.argv) < 2:
    print("Usage: python main.py \"query goes heeere.\"")
    sys.exit(1)

query = sys.argv[1]

data_path = "data/clean.csv"
data = pd.read_csv(data_path, delimiter=';')
descriptions = data['description'].to_list()
names = data['name'].to_list()

vectorizer = Vectorizer()
description_vectors = vectorizer.fit_transform(descriptions)
similarity_search = SimilaritySearch(names, descriptions, description_vectors)

query_vector = vectorizer.transform(query)

top_matches = similarity_search.find_similar(query_vector)

print(top_matches)
