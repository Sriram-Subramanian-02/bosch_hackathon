import os

import cohere
import concurrent.futures

from caching import semantic_cache
from constants import USER_ID, SESSION_ID, COHERE_API_KEY_TEXT
from databases.MongoDB.utils import get_latest_data
from image_processing.services import get_suitable_image
from retrieval_augmented_generation.constants import MAX_DOCS_FOR_CONTEXT
from retrieval_augmented_generation.retriever import normal_retriever
from table_processing.services import reconstruct_table
from text_processing.services import check_probing_conditions
from utils import get_pdf_pages


os.environ["COHERE_API_KEY"] = COHERE_API_KEY_TEXT
semantic_cache = semantic_cache("manual_cache.json")


def get_response(query, threshold=0.35):
    """
    Get a response based on the provided query.

    Args:
        query (str): The user's query.
        threshold (float, optional): The threshold for semantic similarity. Defaults to 0.35.

    Returns:
        tuple: A tuple containing the response text, image ID, PDF pages, DataFrame, and table response.
    """

    chat_history, probing_history = get_latest_data(USER_ID, SESSION_ID)
    print(f"\n\nprobing hist = {probing_history}")

    if len(probing_history) != 0:
        new_query = ''

        for doc in probing_history:
            new_query += f"{doc['query']}. "

        query = new_query + query

        print(f"\n\n\nNew query = {query}\n\n\n")

    else:
        print("\n\n\nNo probing history\n\n\n")

    cache_response, image_ids_from_cache, pdf_pages = semantic_cache.query_cache(query)
    if cache_response is not None:
        return cache_response, image_ids_from_cache, pdf_pages, None, None, None

    context, image_ids, table_data = normal_retriever(query)
    pdf_pages = get_pdf_pages(context)
    context_list = list()
    image_ids = list(set(image_ids))

    for i in context:
        context_list.append(i.page_content)

    co = cohere.Client(api_key=COHERE_API_KEY_TEXT)

    model = "embed-english-v3.0"
    input_type = "search_query"

    query_emb = co.embed(texts=[f"{query}"], model=model, input_type=input_type)

    chat_history = None
    image_id, max_image_content = None, None
    counter = None
    df, table_response = None, None

    # Use ThreadPoolExecutor to run functions in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # future_chat_history = executor.submit(get_latest_data, USER_ID, SESSION_ID)
        future_image_id = executor.submit(
            get_suitable_image, image_ids, query, query_emb
        )
        future_counter = executor.submit(
            check_probing_conditions, context_list, query_emb, threshold
        )
        future_table_response = executor.submit(
            reconstruct_table, table_data, context_list, f"{query}", query_emb
        )

        # chat_history = future_chat_history.result()
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
                If the user doesnot specify the car's name, kindly ask for it as a probing question(available car names are HYUNDAI EXTER or EXTER, HYUNDAI VERNA or VERNA, TATA PUNCH or PUNCH and TATA NEXON or NEXON).
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
                    If the user doesnot specify the car's name, you should say that this response is for this particular car/cars.
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

    co = cohere.Client(COHERE_API_KEY_TEXT)
    response = co.chat(message=prompt, model="command-r", temperature=0)


    if flag_probe:
        return response.text, None, pdf_pages, None, None, flag_probe
    else:
        semantic_cache.insert_into_cache(
            query, query_emb, response.text, image_id, pdf_pages
        )
        return response.text, image_id, pdf_pages, df, table_response, flag_probe


def generalize_image_summary(response):
    """
    Generate a generalized and user-friendly summary of an image description.

    This function uses the Cohere API to transform a detailed image summary into a polite, 
    easy-to-understand bullet-point list, suitable for helping users comprehend car owner manuals.

    Args:
        response (str): The detailed image summary from Qdrant.

    Returns:
        str: A string containing the generalized and user-friendly image summary.
    """
    
    prompt = f"""
        You are a chatbot designed to assist users in understanding car owner manuals.
        Please transform the following image summary into a polite, easy-to-understand list of bullet points.
        Avoid using technical jargon and aim for clarity and simplicity.

        Image Summary: {response}
    """

    co = cohere.Client(COHERE_API_KEY_TABLES)
    response = co.chat(message=prompt, model="command-r", temperature=0)

    return response.text
