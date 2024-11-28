import sys
from pymongo import MongoClient
from src.similarity_search import SentenceSimilaritySearch
from src.vectorizer import Vectorizer
from os import system
import matplotlib.pyplot as plt


def plot_enneagram(vocabulary, min_count):
    for degree in range(2, 5):  
        word_subsets = {}
        for sentence in vocabulary:
            words = sentence.split()
            n = len(words)
            for j in range(n - degree + 1):
                subset = ' '.join(words[j:j + degree])
                if subset in word_subsets:
                    word_subsets[subset] += 1
                else:
                    word_subsets[subset] = 1

        subsets = sorted([subset for subset, count in word_subsets.items() if count > min_count])
        counts = [count for subset, count in sorted(word_subsets.items()) if count > min_count]
    
        if len(subsets) == 0:
            plt.figure(figsize=(12, 6))
            plt.title(f'Enneagram of Degree {degree}', fontsize=12)
            plt.axis('off')
            plt.show()
            continue

        sorted_subsets = [subset for _, subset in sorted(zip(counts, subsets), reverse=True)]
        sorted_counts = sorted(counts, reverse=True)

        plt.figure(figsize=(7, 12))
        plt.barh(sorted_subsets, sorted_counts)
        plt.title(f'Enneagram of Degree {degree}', fontsize=8)
        plt.ylabel('Subsets', fontsize=8)
        plt.xlabel('Counts', fontsize=8)
        plt.tick_params(axis='y', labelrotation=0, labelsize=9.5)
        plt.tight_layout()
        #plt.subplots_adjust(top=1, bottom=0)
        plt.show()

        with open(f'degree_{degree}_subsets.txt', 'w') as f:
            for subset, count in  sorted(word_subsets.items()):
                f.write(f'{subset}: {count}\n')

def load_data_from_mongodb(client):
    
    db = client['directory']
    collection = db['lidia']
    data = list(collection.find())
    descriptions = [doc['description'] for doc in data]
    names = [doc['name'] for doc in data]

    return names, descriptions

if __name__ == "__main__":
    client = MongoClient('mongodb://alex:Z@localhost:27017/?authSource=admin', 27017)
    if len(sys.argv) < 2:
        print("Usage: python main.py \"query goes here.\"")
        sys.exit(1)

    query = sys.argv[1]
    names, descriptions = load_data_from_mongodb(client)

    vectorizer = Vectorizer()
    similarity_search = SentenceSimilaritySearch(names, descriptions, vectorizer, mongo_client=client)
    print("bag of words", len(vectorizer.vocabulary))
    #print(vectorizer.vocabulary)
    top_matches = similarity_search.find_similar_sentences(query, top_n=10)
    top_matches['Sentence'] = top_matches['Sentence'].str.replace(',','')
    top_matches.to_csv("results.csv", sep=",")
    system("column -s, -t < results.csv")
    #plot_enneagram(vectorizer.sentences_split, 0)
    