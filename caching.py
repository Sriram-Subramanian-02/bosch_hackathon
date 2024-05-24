import json
import faiss

import numpy as np
import cohere
from langchain.embeddings.cohere import CohereEmbeddings
from sentence_transformers import SentenceTransformer

from constants import COHERE_API_KEY

def init_cache():
    index = faiss.IndexFlatL2(1024)
    return index


def retrieve_cache(json_file):
    try:
        with open(json_file, 'r') as file:
            cache = json.load(file)
    except FileNotFoundError:
        cache = {'questions': [], 'embeddings': [], 'response_text': []}

    return cache


def store_cache(json_file, cache):
    with open(json_file, 'w') as file:
        json.dump(cache, file)


class semantic_cache:
    def __init__(self, json_file="cache_file.json", threshold=0.15):
        self.index = init_cache()
        self.euclidean_threshold = threshold

        self.json_file = json_file
        self.cache = retrieve_cache(self.json_file)


    def query_cache(self, query):
        try:
            co = cohere.Client(api_key=COHERE_API_KEY)
            model="embed-english-v3.0"
            input_type="search_query"

            query_emb = co.embed(texts=[f"{query}"],
                        model=model,
                        input_type=input_type)
            
            query_emb = np.array(query_emb.embeddings)
            
            D, I = self.index.search(query_emb, 1) 

            if D[0] >= 0:
                if I[0][0] >= 0 and D[0][0] <= self.euclidean_threshold:
                    row_id = int(I[0][0])
                    print('Answer recovered from Cache. ')
                    print('response_text: ' + self.cache['response_text'][row_id])

                    return self.cache['response_text'][row_id]
                return None
            return None
                
        except Exception as e:
            raise RuntimeError(f"Error during querying cache: {e}")
        
    def insert_into_cache(self, query, query_embedding, response_text):
        self.cache['questions'].append(query)
        self.cache['embeddings'].append(query_embedding.embeddings)
        self.cache['response_text'].append(response_text)

        self.index.add(np.array(query_embedding.embeddings))
        store_cache(self.json_file, self.cache)

