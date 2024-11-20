import sys
import pandas as pd
from pymongo import MongoClient
from src.similarity_search import SimilaritySearch
from src.vectorizer import Vectorizer

def load_data_from_mongodb():
    # Connect to MongoDB
    client = MongoClient('mongodb://alex:Z@localhost:27017/?authSource=admin', 27017)
    db = client['directory']  # Replace with your database name
    collection = db['lidia']  # Replace with your collection name

    # Retrieve data from MongoDB
    data = list(collection.find())
    descriptions = [doc['description'] for doc in data]
    names = [doc['name'] for doc in data]

    return names, descriptions

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py \"query goes here.\"")
        sys.exit(1)

    query = sys.argv[1]

    # Load data from MongoDB
    names, descriptions = load_data_from_mongodb()

    vectorizer = Vectorizer()
    description_vectors = vectorizer.fit_transform(descriptions)

    # Create SimilaritySearch instance with the vectorizer
    similarity_search = SimilaritySearch(names, descriptions, description_vectors, vectorizer)

    query_vector = vectorizer.transform(query)

    top_matches = similarity_search.find_similar(query)

    print(top_matches)
