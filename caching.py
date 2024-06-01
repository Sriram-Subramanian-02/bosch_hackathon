import json
import faiss

import numpy as np
import cohere
from langchain.embeddings.cohere import CohereEmbeddings
from sentence_transformers import SentenceTransformer

from constants import COHERE_API_KEY_TABLES


def init_cache():
    """
    Initialize a Faiss index for caching semantic embeddings.

    Returns:
        faiss.IndexFlatL2: Initialized Faiss index.
    """

    index = faiss.IndexFlatL2(1024)
    return index


def retrieve_cache(json_file):
    """
    Retrieve the cache from a JSON file if it exists, otherwise initialize an empty cache.

    Args:
        json_file (str): Path to the JSON file containing the cache.

    Returns:
        dict: Cache dictionary containing lists for questions, embeddings, response texts, and image IDs.
    """

    try:
        with open(json_file, "r") as file:
            cache = json.load(file)
    except FileNotFoundError:
        cache = {
            "questions": [],
            "embeddings": [],
            "response_text": [],
            "image_id": [],
            "pdf_pages": [],
        }

    return cache


def store_cache(json_file, cache):
    """
    Store the cache dictionary into a JSON file.

    Args:
        json_file (str): Path to the JSON file to store the cache.
        cache (dict): Cache dictionary to be stored.
    """

    with open(json_file, "w") as file:
        json.dump(cache, file)


class semantic_cache:
    """
    A class to manage a semantic cache for storing and retrieving queries and their corresponding embeddings.

    Attributes:
        index (faiss.IndexFlatL2): Faiss index for semantic embeddings.
        euclidean_threshold (float): Threshold for Euclidean distance for retrieval.
        json_file (str): Path to the JSON file containing the cache.
        cache (dict): Cache dictionary containing lists for questions, embeddings, response texts, and image IDs.
    """

    def __init__(self, json_file="cache_file.json", threshold=0.15):
        """
        Initialize the semantic cache.

        Args:
            json_file (str, optional): Path to the JSON file containing the cache. Defaults to "cache_file.json".
            threshold (float, optional): Threshold for Euclidean distance for retrieval. Defaults to 0.15.
        """

        self.index = init_cache()
        self.euclidean_threshold = threshold

        self.json_file = json_file
        self.cache = retrieve_cache(self.json_file)

    def query_cache(self, query):
        """
        Query the cache for a given query.

        Args:
            query (str): Query string.

        Returns:
            tuple or None: Tuple containing response text and image ID if a match is found within the threshold, otherwise None.
        """

        try:
            co = cohere.Client(api_key=COHERE_API_KEY_TABLES)
            model = "embed-english-v3.0"
            input_type = "search_query"

            query_emb = co.embed(texts=[f"{query}"], model=model, input_type=input_type)

            query_emb = np.array(query_emb.embeddings)

            D, I = self.index.search(query_emb, 1)

            if D[0] >= 0:
                if I[0][0] >= 0 and D[0][0] <= self.euclidean_threshold:
                    row_id = int(I[0][0])
                    print("Answer recovered from Cache. ")
                    print("response_text: " + self.cache["response_text"][row_id])

                    return (
                        self.cache["response_text"][row_id],
                        self.cache["image_id"][row_id],
                        self.cache["pdf_pages"][row_id],
                    )
                return None, None, None
            return None, None, None

        except Exception as e:
            raise RuntimeError(f"Error during querying cache: {e}")

    def insert_into_cache(
        self, query, query_embedding, response_text, image_id, pdf_pages
    ):
        """
        Insert a new entry into the cache.

        Args:
            query (str): Query string.
            query_embedding: Query embedding object.
            response_text (str): Response text corresponding to the query.
            image_id (str): Image ID corresponding to the query.
        """

        self.cache["questions"].append(query)
        self.cache["embeddings"].append(query_embedding.embeddings)
        self.cache["response_text"].append(response_text)
        self.cache["image_id"].append(image_id)
        self.cache["pdf_pages"].append(pdf_pages)

        self.index.add(np.array(query_embedding.embeddings))
        store_cache(self.json_file, self.cache)
