import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SentenceSimilaritySearch:
    def __init__(self, names, documents, vectorizer, mongo_client=None):
        # MongoDB connection for synonyms and antonyms
        self.client = mongo_client
        self.db = self.client['directory'] if self.client else None
        self.synonyms_collection = self.db['sinonimos'] if self.db is not None else None
        self.antonyms_collection = self.db['antonimos'] if self.db is not None else None
        
        # Core similarity search setup
        self.names = names
        self.vectorizer = vectorizer
        self.sentence_vectors, self.sentences = vectorizer.fit_transform(documents)
        
        # Create mapping of names to their full document sentences
        self.document_sentences = {}
        self.sentence_to_name = []
        for i, name in enumerate(names):
            documents_sentences = vectorizer.split_sentences(documents[i])
            self.document_sentences[name] = documents_sentences
            self.sentence_to_name.extend([name] * len(documents_sentences))
    
    def _get_semantic_variants(self, tokens):
        """
        Expand tokens with synonyms and handle antonyms
        
        Args:
            tokens (list): Original tokens from the query
        
        Returns:
            dict: Contains expanded tokens and tokens to exclude
        """
        # If no MongoDB connection, return original tokens
        if not self.client:
            return {
                'tokens': tokens,
                'excluded_tokens': []
            }
        
        expanded_tokens = set(tokens)
        excluded_tokens = set()
        
        for token in tokens:
            # Find synonyms
            synonym_doc = self.synonyms_collection.find_one({"palabra": token})
            if synonym_doc and 'sinonimos' in synonym_doc:
                expanded_tokens.update(synonym_doc['sinonimos'])
            
            # Find and mark antonyms
            antonym_doc = self.antonyms_collection.find_one({"palabra": token})
            if antonym_doc and 'antonimos' in antonym_doc:
                # Add antonyms to excluded tokens
                excluded_tokens.update(antonym_doc['antonimos'])
        
        return {
            'tokens': list(expanded_tokens),
            'excluded_tokens': list(excluded_tokens)
        }
    
    def find_similar_sentences(self, query, top_n=20):
        # Preprocess tokens
        tokens = self.vectorizer.preprocess(query)
        
        # Get semantic variants
        semantic_variants = self._get_semantic_variants(tokens)
        expanded_tokens = set(self.vectorizer.preprocess(" ".join(semantic_variants['tokens'])))
        excluded_tokens = set(self.vectorizer.preprocess(" ".join(semantic_variants['excluded_tokens'])))
        
        print("Original tokens:", tokens)
        print("Expanded tokens:", expanded_tokens)
        print("Excluded tokens:", excluded_tokens)
        
        # Generate query vector using expanded tokens
        expanded_query = " ".join(expanded_tokens)
        query_vector = self.vectorizer.transform_sentence(expanded_query).reshape(1, -1)
        
        # Compute similarities
        similarities = cosine_similarity(query_vector, self.sentence_vectors).flatten()
        
        # Create results DataFrame
        results = pd.DataFrame({
            'Name': self.sentence_to_name,
            'Sentence': self.sentences,
            'Similarity': similarities
        })
        
        # Exclude entire descriptions if any sentences contain excluded tokens
        if excluded_tokens:
            # Identify names with antonym-containing sentences
            names_to_exclude = set()
            for name, sentences in self.document_sentences.items():
                for sentence in sentences:
                    if any(
                        excluded_token in sentence.lower() 
                        for excluded_token in map(str.lower, excluded_tokens)
                    ):
                        names_to_exclude.add(name)
                        break  # Once we find an antonym, exclude the entire description
            
            # Filter out results from names with antonyms
            results = results[~results['Name'].isin(names_to_exclude)]
        
        # Filter and aggregate results
        results = results[results['Similarity'] > 0]
        aggregated_results = results.groupby('Name', as_index=False).agg({
            'Similarity': 'sum'
        })
        aggregated_results = aggregated_results.sort_values(by='Similarity', ascending=False)
        
        # Merge and sort results
        results = results.merge(aggregated_results[['Name', 'Similarity']], on='Name', suffixes=('', '_aggregated'))
        results = results.sort_values(by='Similarity', ascending=False)
        
        # Order results by name
        ordered_results = []
        for name in aggregated_results['Name']:
            person_results = results[results['Name'] == name]
            ordered_results.append(person_results[['Name', 'Sentence', 'Similarity']])
        
        final_results = pd.concat(ordered_results)
        return final_results.head(top_n)