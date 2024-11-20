import sys
import pandas as pd
from pymongo import MongoClient
from spacy.tokens.token import Token
from src.similarity_search import SentenceSimilaritySearch
from src.vectorizer import Vectorizer
from os import system

def load_data_from_mongodb():
    client = MongoClient('mongodb://alex:Z@localhost:27017/?authSource=admin', 27017)
    db = client['directory']
    collection = db['lidia']
    data = list(collection.find())
    descriptions = [doc['description'] for doc in data]
    names = [doc['name'] for doc in data]

    return names, descriptions

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py \"query goes here.\"")
        sys.exit(1)

    query = sys.argv[1]
    names, descriptions = load_data_from_mongodb()

    vectorizer = Vectorizer()
    similarity_search = SentenceSimilaritySearch(names, descriptions, vectorizer)

    top_matches = similarity_search.find_similar_sentences(query, top_n=10)
    pd.set_option('display.max_colwidth', None)
    top_matches['Sentence'] = top_matches['Sentence'].str.replace(',','')
    top_matches.to_csv("results.csv", sep=",")
    system("column -s, -t < results.csv")
