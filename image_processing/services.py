import time

import cohere
from qdrant_client import models, QdrantClient
from sentence_transformers.util import cos_sim

from constants import COHERE_API_KEY_IMAGES
from databases.QDrant.constants import (
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
)


def return_images_context(image_ids):
    """
    Retrieves image context information from the Qdrant database based on image IDs.

    This function queries the Qdrant collection to retrieve documents that match the given
    image IDs and have the chunk type "Image". It returns a dictionary mapping the page content
    to the corresponding image IDs.

    Args:
        image_ids (List[str]): A list of image IDs to search for in the Qdrant collection.

    Returns:
        Dict[str, List[str]]: A dictionary where keys are the page content and values are lists of image IDs.
    """

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

    must_filters = [
        models.FieldCondition(
            key="metadata.chunk_type", match=models.MatchValue(value="Image")
        )
    ]

    for i in qdrant_client.scroll(
        collection_name=f"{QDRANT_COLLECTION_NAME}",
        scroll_filter=models.Filter(should=should_filters, must=must_filters),
        limit=100,
    )[0]:
        text_to_image_ids[i.payload["page_content"]] = i.payload["metadata"][
            "image_ids"
        ]

    return text_to_image_ids


def get_suitable_image(image_ids, query, query_emb, img_threshold=0.3):
    """
    Finds the most suitable image based on the query and its embedding.

    This function retrieves image contexts using the given image IDs, and then uses an LLM
    to find the context most similar to the query. It checks if the car names match between
    the query and context keys, and returns the value of the most similar context key if it
    meets the similarity threshold.

    Args:
        image_ids (List[str]): A list of image IDs to search for in the Qdrant collection.
        query (str): The input query string.
        query_emb: The embedding of the query for similarity comparison.
        img_threshold (float): The similarity threshold to consider for choosing the image. Default is 0.3.

    Returns:
        Tuple[str, str]: A tuple containing:
            - str: The image ID of the most suitable image.
            - str: The context text associated with the most suitable image.
            If no suitable image is found, returns (None, None).
    """

    text_to_image_ids = return_images_context(image_ids)
    # for i in text_to_image_ids:
        # print(i)
        # print(text_to_image_ids[i])
    # new_text_to_image_ids = dict()
    images_context_values = list(text_to_image_ids.keys())

    co = cohere.Client(api_key=COHERE_API_KEY_IMAGES)

    model = "embed-english-v3.0"
    input_type = "search_query"

    prompt = f"""
        Given a dictionary: {text_to_image_ids} and string: {query}, choose the key of the dictionary that is almost as close as possible to the string and return the value of the key in the dictionary.
        Note: Check for similarity between the provided string and keys of the dictionary and only return the value of the key that is similar to the string.        
        Choose the key of dictionary that is highly similar and closer to the string provided so that I can get the value of the key and display that image in the UI.
        Extract the car name from the string and then extract the car name from keys of the dictionary. Only give the value of the key when car names are same in both.
        Return only the value of the key choosen. I do not need anything else.
    """
    time.sleep(1)
    co = cohere.Client(COHERE_API_KEY_IMAGES)
    response = co.chat(message=prompt, model="command-r", temperature=0)
    print("\n\n")
    print(f"answer from llm is: {response.text}")
    print("\n\n")

    max_image_context = None
    time.sleep(1)
    for key, value in text_to_image_ids.items():
        if value == str(response.text):
            max_image_context = key

    if not max_image_context:
        d = return_images_context([str(response.text)])
        if d:
            max_image_context = list(d.keys())[0]

    print(f"\n\nmax_image content is: \n{max_image_context}")

    if max_image_context is None:
        return None, None

    image_emb = co.embed(texts=[max_image_context], model=model, input_type=input_type)
    val = float(cos_sim(query_emb.embeddings, image_emb.embeddings)[0][0])

    print(f"\n\n\nImage similarity value is {val}\n\n\n")
    if val >= img_threshold:
        return str(response.text), max_image_context

    else:
        return None, None
