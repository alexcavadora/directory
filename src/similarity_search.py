import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class SentenceSimilaritySearch:
    def __init__(self, names, documents, vectorizer):
        self.names = names
        self.vectorizer = vectorizer
        self.sentence_vectors, self.sentences = vectorizer.fit_transform(documents)
        self.sentence_to_name = []
        for i, doc in enumerate(documents):
            sentences = vectorizer.split_sentences(doc)
            self.sentence_to_name.extend([names[i]] * len(sentences))

    def find_similar_sentences(self, query, top_n=10):
        query_vector = self.vectorizer.transform_sentence(query).reshape(1, -1)
        print("tokens: ",self.vectorizer.preprocess(query))
        similarities = cosine_similarity(query_vector, self.sentence_vectors).flatten()
        results = pd.DataFrame({
            'Name': self.sentence_to_name,
            'Sentence': self.sentences,
            'Similarity': similarities
        })

        results = results[results['Similarity'] > 0]

        aggregated_results = results.groupby('Name', as_index=False).agg(
            {'Similarity': 'sum'}
        )
        aggregated_results = aggregated_results.sort_values(by='Similarity', ascending=False)

        results = results.merge(aggregated_results[['Name', 'Similarity']], on='Name', suffixes=('', '_aggregated'))
        results = results.sort_values(by='Similarity', ascending=False)
        ordered_results = []

        for name in aggregated_results['Name']:
            person_results = results[results['Name'] == name]
            ordered_results.append(person_results[['Name', 'Sentence', 'Similarity']])
        final_results = pd.concat(ordered_results)

        return final_results.head(top_n)
