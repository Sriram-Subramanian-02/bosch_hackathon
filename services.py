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
import json
from qdrant_client import models, QdrantClient
import cohere
import numpy as np
from qdrant_client.http.models import Batch
import concurrent.futures
from io import BytesIO
from pdf2image import convert_from_path
import streamlit.components.v1 as components
import base64
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from PIL import Image
import torch
import io


from caching import semantic_cache
from constants import USER_ID, SESSION_ID, QDRANT_API_KEY, QDRANT_URL, QDRANT_COLLECTION_NAME,COHERE_API_KEY_2, COHERE_API_KEY_1
from utils import get_latest_data


os.environ["COHERE_API_KEY"] = COHERE_API_KEY_2
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
    table_data = list()
    from itertools import chain
    for document in result:
        image_ids.append(document.metadata['image_ids'])
        if document.metadata['chunk_type'] == 'Table':
            table_data.append(document.page_content)
    image_ids = list(chain.from_iterable([item] if isinstance(item, str) else item for item in image_ids if item is not None))

    print(image_ids)

    # if len(table_data) > 0:
    #     table_data = str(table_data[0])
    #     return result, image_ids, table_data
    
    # else:
    return result, image_ids, table_data


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
    co = cohere.Client(api_key=COHERE_API_KEY_2)
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


def load_clip_model(file_path = "models"):
    model_ID = "openai/clip-vit-base-patch32"

    model = CLIPModel.from_pretrained(f"{file_path}/clip_model")
    processor = CLIPProcessor.from_pretrained(model_ID)

    return model, processor


def get_image_context_from_QDrant(image_vector):
        qdrant_client = QdrantClient(
            "https://35ebdc7d-ec99-4ebd-896c-ff5705cf369b.us-east4-0.gcp.cloud.qdrant.io:6333",
            prefer_grpc=True,
            api_key="9dKJsKOYwT0vGlWPrZXBSIlbUzvRdJ1XkM0_floo8FmYCOHX_Y0y-Q",
        )

        search_result = qdrant_client.search(
            collection_name="owners_manual_images",
            query_vector=image_vector[0].tolist(),
            limit = 1
        )

        payloads = [hit.payload for hit in search_result]
        print(payloads)
        image_id = payloads[0]['metadata']['image_id']
        image_context = return_images_context([image_id])
        image_summary = list(image_context.keys())[0]
        print(image_summary)
        return image_summary


def get_image_summary(image_bytes, image_format):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading Clip model")
    start_time = time.time()
    model, processor = load_clip_model()
    end_time = time.time()
    execution_time = end_time - start_time
    print("Model Loaded")
    print(f"\n\nExecution time for loading the model: {execution_time} seconds")

    input_image_directory_path = "input_data/user_image_input"
    if not os.path.exists(input_image_directory_path):
        os.makedirs(input_image_directory_path, exist_ok=True)
        
    image_path = f"input_data/user_image_input/input_image.{image_format}"
    with open(image_path, 'wb') as f:
        f.write(image_bytes)

    my_image = Image.open(image_path)

    image = processor(
        text=None,
        images=my_image,
        return_tensors="pt"
    )["pixel_values"].to(device)

    # Get the image features
    embedding = model.get_image_features(image)
    embedding_as_np = embedding.cpu().detach().numpy()
    return get_image_context_from_QDrant(embedding_as_np)


def get_pdf_pages(context):
    pdf_pages = {}
    for doc in context:
        car_name = doc.metadata['car_name']
        if car_name not in pdf_pages:
            pdf_pages[car_name] = [doc.metadata['page_number']]
        else:
            pdf_pages[car_name].append(doc.metadata['page_number'])

    return pdf_pages
    

def pdf_to_images(pdf_path, pages=None):
    images = []
    for page_num in pages:
        images.extend(convert_from_path(pdf_path, first_page=page_num+1, last_page=page_num+1))

    image_paths = []
    for image in images:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        image_paths.append(img_str)
    return image_paths


def reconstruct_table(table_data, context, query, query_emb, table_threshold=0.5):
    model="embed-english-v3.0"
    input_type="search_query"

    
    prompt = f"""
        Reconstruct the table using this table data: {table_data}, question: {query} and context: {context}.
        Reconstruct only one element of table_data that is most similar to the question.
        Return the table in json format and do not add anything else like new line characters or tab spaces.
        If car names in question and table_data does not match return empty string.
        If you feel that json format that you are returning is not similar to question, return empty string.
    """

    co = cohere.Client(COHERE_API_KEY_2)
    response = co.chat(
        message=prompt,
        model="command-r",
        temperature=0
    )
    response = response.text
    print(response)
    if response == '' or response is None or response == 'None':
        return None, None
    formatted_response = response.replace('\n', '').replace('\t', '').replace('`', '').replace('json', '')
    print(formatted_response)
    try:
        data = json.loads(formatted_response)
        json_string = json.dumps(data)

        table_emb = co.embed(texts=[json_string],
                    model=model,
                    input_type=input_type)
        val = float(cos_sim(query_emb.embeddings, table_emb.embeddings)[0][0])
        print(val)
        if val < table_threshold:
            return None, None

        print(json.dumps(data, indent=4))
        try:
            df = pd.DataFrame(data)

            return df, formatted_response
    
        except:
            return None, data
        
    except:
        return None, formatted_response


def get_response(query, threshold=0.35):
    # chat_history = get_latest_data(USER_ID, SESSION_ID)
    
    cache_response, image_ids_from_cache = semantic_cache.query_cache(query)
    if cache_response is not None:
        return cache_response, image_ids_from_cache, None, None, None
    
    context, image_ids, table_data = normal_retriever(query)
    pdf_pages = get_pdf_pages(context)
    context_list = list()
    image_ids = list(set(image_ids))

    for i in context:
        context_list.append(i.page_content)

    co = cohere.Client(api_key=COHERE_API_KEY_2)

    model="embed-english-v3.0"
    input_type="search_query"

    query_emb = co.embed(texts=[f"{query}"],
                model=model,
                input_type=input_type)
    
    chat_history = None
    image_id, max_image_content = None, None
    counter = None
    df, table_response = None, None

    # Use ThreadPoolExecutor to run functions in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_chat_history = executor.submit(get_latest_data, USER_ID, SESSION_ID)
        future_image_id = executor.submit(get_suitable_image, image_ids, query, query_emb)
        future_counter = executor.submit(check_probing_conditions, context_list, query_emb, threshold)
        future_table_response = executor.submit(reconstruct_table, table_data, context_list, f"{query}", query_emb)

        
        chat_history = future_chat_history.result()
        image_id, max_image_content = future_image_id.result()
        df, table_response = future_table_response.result()
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
            

    co = cohere.Client(COHERE_API_KEY_2)
    response = co.chat(
        message=prompt,
        model="command-r",
        temperature=0
    )

    semantic_cache.insert_into_cache(query, query_emb, response.text, image_id)

    if flag_probe:
        return response.text, None, pdf_pages, None, None
    else:
        return response.text, image_id, pdf_pages, df, table_response



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
