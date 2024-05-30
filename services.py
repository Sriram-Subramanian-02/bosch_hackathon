import os
from operator import itemgetter
from langchain.load import dumps, loads
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.llms import Cohere
import numpy as np
import time
from sentence_transformers.util import cos_sim
from langchain_core.documents.base import Document
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
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
import concurrent.futures


from caching import semantic_cache
from constants import USER_ID, SESSION_ID, QDRANT_API_KEY, QDRANT_URL, QDRANT_COLLECTION_NAME,COHERE_API_KEY_2, COHERE_API_KEY_1
from utils import get_latest_data


os.environ["COHERE_API_KEY"] = COHERE_API_KEY_1
TOP_K = 10
MAX_DOCS_FOR_CONTEXT = 10
semantic_cache = semantic_cache('manual_cache.json')

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


def reciprocal_rank_fusion(results: list[list], k=60):
    """Rerank docs (Reciprocal Rank Fusion)

    Args:
        results (list[list]): retrieved documents
        k (int, optional): parameter k for RRF. Defaults to 60.

    Returns:
        ranked_results: list of documents reranked by RRF
    """

    # print("\n\n\nresults in rrf function: ", len(results))

    fused_scores = {}
    for docs in results:
        # Assumes the docs are returned in sorted order of relevance
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # return only documents
    return [x[0] for x in reranked_results[:MAX_DOCS_FOR_CONTEXT]]


def query_generator(original_query: dict) -> list[str]:
    """Generate queries from original query

    Args:
        query (dict): original query

    Returns:
        list[str]: list of generated queries
    """

    # original query
    query = original_query.get("query")

    # prompt for query generator
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that generates multiple search queries based on a single input query."),
        ("user", "Generate multiple search queries related to:  {original_query}. When creating queries, please refine or add closely related contextual information, without significantly altering the original query's meaning"),
        ("user", "OUTPUT (3 queries):")
    ])

    # LLM model
    model = Cohere()

    # query generator chain
    query_generator_chain = (
        prompt | model | StrOutputParser() | (lambda x: x.split("\n"))
    )

    # gererate queries
    queries = query_generator_chain.invoke({"original_query": query})

    # add original query
    queries.insert(0, "0. " + query)

    return queries


def rrf_retriever(query: str) -> list[Document]:
    """RRF retriever

    Args:
        query (str): Query string

    Returns:
        list[Document]: retrieved documents
    """

    # Retriever
    embedding = CohereEmbeddings(model = "embed-english-v3.0")
    
    qdrant_client = QdrantClient(
        QDRANT_URL,
        prefer_grpc=True,
        api_key=QDRANT_API_KEY,
    )

    qdrant = Qdrant(
        client=qdrant_client,
        collection_name=QDRANT_COLLECTION_NAME,
        embeddings=embedding,
    )

    retriever = qdrant.as_retriever(
        search_kwargs={'k': TOP_K},
        metadata={"car_name": "Hyundai Exter"}
    )

    # RRF chain
    chain = (
        {"query": itemgetter("query")}
        | RunnableLambda(query_generator)
        | retriever.map()
        | reciprocal_rank_fusion
    )

    # invoke
    result = chain.invoke({"query": query})

    image_ids = []
    from itertools import chain
    for document in result:
        image_ids.append(document.metadata['image_ids'])
    image_ids = list(chain.from_iterable([item] if isinstance(item, str) else item for item in image_ids if item is not None))

    print(image_ids)
    return result, image_ids


def calculate_similarity(a, b):
  return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def normal_retriever(query: str) -> list[Document]:
    # Retriever
    embedding = CohereEmbeddings(model = "embed-english-v3.0")
    
    qdrant_client = QdrantClient(
        QDRANT_URL,
        prefer_grpc=True,
        api_key=QDRANT_API_KEY,
    )
    print(qdrant_client)

    qdrant = Qdrant(
        client=qdrant_client,
        collection_name=QDRANT_COLLECTION_NAME,
        embeddings=embedding,
    )

    retriever = qdrant.as_retriever(
        search_kwargs={'k': TOP_K},
        metadata={}
    )

    # invoke
    result = retriever.invoke(query)
    print(result)

    image_ids = []
    from itertools import chain
    for document in result:
        image_ids.append(document.metadata['image_ids'])
    image_ids = list(chain.from_iterable([item] if isinstance(item, str) else item for item in image_ids if item is not None))

    print(image_ids)

    return result, image_ids


def return_images_context(image_ids):
    text_to_image_ids = dict()

    qdrant_client = QdrantClient(
        QDRANT_URL,
        prefer_grpc=True,
        api_key=QDRANT_API_KEY,
    )

    should_filters = list()

    for i in image_ids:
        should_filters.append(
            models.FieldCondition(
                    key="metadata.image_ids",
                    match=models.MatchValue(value=i),
                )
        )

    must_filters=[models.FieldCondition(key="metadata.chunk_type", match=models.MatchValue(value="Image"))]

    for i in qdrant_client.scroll(collection_name=f"{QDRANT_COLLECTION_NAME}", scroll_filter=models.Filter(should=should_filters, must=must_filters),limit=100)[0]:
        text_to_image_ids[i.payload['page_content']] = i.payload['metadata']['image_ids']

    return text_to_image_ids


def check_probing_conditions(context_list, query_emb, threshold):
    co = cohere.Client(api_key=COHERE_API_KEY_1)
    model="embed-english-v3.0"
    input_type="search_query"

    time.sleep(1)
    res = co.embed(texts=context_list,
                    model=model,
                    input_type=input_type)

    counter = 0
    for i in res.embeddings:
        if float(cos_sim(query_emb.embeddings, i)[0][0]) < threshold:
            print(float(cos_sim(query_emb.embeddings, i)[0][0]))
            counter += 1

    print(f"counter = {counter}")

    return counter



def get_suitable_image(image_ids, query, query_emb, img_threshold=0.3):
    text_to_image_ids = return_images_context(image_ids)
    for i in text_to_image_ids:
        print(i)
        print(text_to_image_ids[i])
    # new_text_to_image_ids = dict()
    images_context_values = list(text_to_image_ids.keys())

    co = cohere.Client(api_key=COHERE_API_KEY_2)

    model="embed-english-v3.0"
    input_type="search_query"

    # for i in images_context_values:
    #     # print(i)
    #     image_emb = co.embed(texts=[i],
    #                 model=model,
    #                 input_type=input_type)
    #     val = float(cos_sim(query_emb.embeddings, image_emb.embeddings)[0][0])
    #     if val >= img_threshold:
    #         new_text_to_image_ids[i] = text_to_image_ids[i]
            # print(i)
            # print(val)
            # print(new_text_to_image_ids[i])
            # print("\n")

    prompt = f"""
        Given a dictionary: {text_to_image_ids} and string: {query}, choose the key of the dictionary that is almost as close as possible to the string and return the value of the key in the dictionary.
        Note: Check for similarity between the provided string and keys of the dictionary and only return the value of the key that is similar to the string.        
        Choose the key of dictionary that is highly similar and closer to the string provided so that I can get the value of the key and display that image in the UI.
        Extract the car name from the string and then extract the car name from keys of the dictionary. Only give the value of the key when car names are same in both.
        Return only the value of the key choosen. I do not need anything else.
    """
    time.sleep(1)
    co = cohere.Client(COHERE_API_KEY_2)
    response = co.chat(
        message=prompt,
        model="command-r",
        temperature=0
    )
    print("\n\n")
    print(f"answer from llm is: {response.text}")
    print("\n\n")


    
    max_image_context = None
    time.sleep(1)
    for key, value in text_to_image_ids.items():
        if value == str(response.text):
            max_image_context = key

    # print(f"max_image content is: \n{max_image_context}")

    if max_image_context is None:
        return None, None
    
    image_emb = co.embed(texts=[max_image_context],
                model=model,
                input_type=input_type)
    val = float(cos_sim(query_emb.embeddings, image_emb.embeddings)[0][0])

    print(f"Image similarity value is {val}")
    if val >= img_threshold:
        # print("hi there")
        return str(response.text), max_image_context

    else:
        return None, None
        
    

    # print(f"\n\n{max_image_context}")

    # max_image_context = max(text_to_scores, key=text_to_scores.get)
    # image_id = text_to_image_ids[max_image_context]

    # for i in text_to_image_ids:
    #     print(i)
    #     print(text_to_image_ids[i])
    #     print("\n\n")

    

def get_response(query, threshold=0.35):
    # chat_history = get_latest_data(USER_ID, SESSION_ID)
    
    cache_response, image_ids_from_cache = semantic_cache.query_cache(query)
    if cache_response is not None:
        return cache_response, image_ids_from_cache
    
    context, image_ids = normal_retriever(query)
    context_list = list()
    image_ids = list(set(image_ids))

    for i in context:
        context_list.append(i.page_content)

    co = cohere.Client(api_key=COHERE_API_KEY_1)

    model="embed-english-v3.0"
    input_type="search_query"

    query_emb = co.embed(texts=[f"{query}"],
                model=model,
                input_type=input_type)
    
    chat_history = None
    image_id, max_image_content = None, None
    counter = None

    # Use ThreadPoolExecutor to run functions in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_chat_history = executor.submit(get_latest_data, USER_ID, SESSION_ID)
        future_image_id = executor.submit(get_suitable_image, image_ids, query, query_emb)
        future_counter = executor.submit(check_probing_conditions, context_list, query_emb, threshold)
        
        chat_history = future_chat_history.result()
        image_id, max_image_content = future_image_id.result()
        counter = future_counter.result()

    prompt = None
    flag_probe = False

    if (MAX_DOCS_FOR_CONTEXT - counter) < 5:
        prompt = f"""
                You are a chatbot built for helping users understand car's owner manuals, try and ask probing questions related to that alone.
                Create several question based on question:{query}, context: {context} and chat history of the user: {chat_history}.
                As similarity between query and context is low, try to ask several probing questions.
                Ask several followup questions to get further clarity.
                Answer in a polite tone, and convey to the user that you need more clarity to answer the question.
                If the user doesnot specify the car's name, kindly ask for it as a probing question(available car names are HYUNDAI EXTER and TATA NEXON).
                Then display the probing questions as bulletin points.
                Do not use technical words, give easy to understand responses.
                If the question asked is a generic question or causal question answer them without using the context.
                If the question is a general question, try to interact with the user in a polite way.
            """
        flag_probe = True
    else:
        if image_id is None:
            prompt = f"""
                    You are a chatbot built for helping users understand car's owner manuals.
                    Answer the question:{query} only based on the context: {context} and the chat history of the user: {chat_history} provided.
                    Try to answer in bulletin points.
                    If the user doesnot specify the car's name, kindly ask for it as a probing question. Else you should say that this response is for this particular car/cars.
                    Do not mention anything about images or figures.
                    Do not use technical words, give easy to understand responses.
                    Do not divulge any other details other than query or context.
                    If the question asked is a generic question or causal question answer them without using the context.
                    If the question is a general question, try to interact with the user in a polite way.
                """
        else:
            prompt = f"""
                    You are a chatbot built for helping users understand car's owner manuals.
                    Answer the question:{query} only based on the context: {context}, the chat history of the user: {chat_history} and this image summary: {max_image_content} provided.
                    Try to answer in bulletin points.
                    Do not use technical words, give easy to understand responses.
                    Do not divulge any other details other than query or context.
                    If the question asked is a generic question or causal question answer them without using the context.
                    If the question is a general question, try to interact with the user in a polite way.
                """
            

    co = cohere.Client(COHERE_API_KEY_1)
    response = co.chat(
        message=prompt,
        model="command-r",
        temperature=0
    )

    semantic_cache.insert_into_cache(query, query_emb, response.text, image_id)

    if flag_probe:
        return response.text, None
    else:
        return response.text, image_id



# def get_response(query, threshold=0.5):
#     cohere_client = cohere.Client(COHERE_API_KEY)
#     chat_history = get_latest_data(USER_ID, SESSION_ID)

#     qdrant_client = QdrantClient(
#         url=QDRANT_URL,
#         api_key=QDRANT_API_KEY,
#     )

#     hits = qdrant_client.search(
#         collection_name="bosch_v1",
#         query_vector=cohere_client.embed(
#             model="embed-english-v3.0",  # New Embed v3 model
#             input_type="search_query",  # Input type for search queries
#             texts=[f"{query}"],
#         ).embeddings[0],
#         limit = 5
#     )

#     result = list()
#     context = list()

#     for hit in hits:
#         result.append({"payload":hit.payload, "score":hit.score})
#         print(hit.score)
#         context.append(hit.payload)

#     counter = 0
#     for i in result:
#         if i["score"] < threshold:
#             counter += 1

#     print(counter)
#     prompt = None

#     if counter >= 3:
#         prompt = f"""
#                 Create several question based on question:{query}, context: {context} and chat history of the user: {chat_history}.
#                 As similarity between query and context is low, try to ask several probing questions.
#                 Ask several followup questions to get further clarity.
#                 Answer in a polite tone, and convey to the user that you need more clarity to answer the question.
#                 Then display the probing questions as bulletin points.
#                 Do not use technical words, give easy to understand responses.
#             """
#     else:
#         prompt = f"""
#                 Answer the question:{query} only based on the context: {context} and the chat history of the user: {chat_history} provided.
#                 Try to answer in bulletin points.
#                 Do not use technical words, give easy to understand responses.
#                 Do not divulge any other details other than query or context.
#             """

#     co = cohere.Client(COHERE_API_KEY)
#     response = co.chat(
#         message=prompt,
#         model="command-r",
#         temperature=0
#     )

#     return response.text
