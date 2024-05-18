import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.llms import Cohere
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.vectorstores import Qdrant
from langchain.document_loaders import TextLoader
import qdrant_client
from langchain.chains import ConversationalRetrievalChain
import pandas as pd
from qdrant_client import models, QdrantClient
import cohere
import numpy as np
from qdrant_client.http.models import Batch


from constants import USER_ID, SESSION_ID, QDRANT_API_KEY, QDRANT_URL, COHERE_API_KEY
from utils import get_latest_data


os.environ["COHERE_API_KEY"] = COHERE_API_KEY


def create_collection(collection_name):
    client = QdrantClient(
        url = QDRANT_URL,
        api_key = QDRANT_API_KEY,
    )

    client.create_collection(
        collection_name = f"{collection_name}",
        vectors_config = models.VectorParams(size=1024, distance=models.Distance.COSINE),
    )


def delete_collection(collection_name):
    client = QdrantClient(
        url = QDRANT_URL,
        api_key = QDRANT_API_KEY,
    )

    client.delete_collection(collection_name = f"{collection_name}")


def get_response(query, threshold=0.5):
    cohere_client = cohere.Client(COHERE_API_KEY)
    chat_history = get_latest_data(USER_ID, SESSION_ID)

    qdrant_client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )

    hits = qdrant_client.search(
        collection_name="bosch_v1",
        query_vector=cohere_client.embed(
            model="embed-english-v3.0",  # New Embed v3 model
            input_type="search_query",  # Input type for search queries
            texts=[f"{query}"],
        ).embeddings[0],
        limit = 5
    )

    result = list()
    context = list()

    for hit in hits:
        result.append({"payload":hit.payload, "score":hit.score})
        print(hit.score)
        context.append(hit.payload)

    counter = 0
    for i in result:
        if i["score"] < threshold:
            counter += 1

    print(counter)
    prompt = None

    if counter >= 3:
        prompt = f"""
                Create several question based on question:{query}, context: {context} and chat history of the user: {chat_history}.
                As similarity between query and context is low, try to ask several probing questions.
                Ask several followup questions to get further clarity.
                Answer in a polite tone, and convey to the user that you need more clarity to answer the question.
                Then display the probing questions as bulletin points.
                Do not use technical words, give easy to understand responses.
            """
    else:
        prompt = f"""
                Answer the question:{query} only based on the context: {context} and the chat history of the user: {chat_history} provided.
                Try to answer in bulletin points.
                Do not use technical words, give easy to understand responses.
                Do not divulge any other details other than query or context.
            """

    co = cohere.Client(COHERE_API_KEY)
    response = co.chat(
        message=prompt,
        model="command-r",
        temperature=0
    )

    return response.text