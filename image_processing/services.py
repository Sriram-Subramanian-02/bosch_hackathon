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
    text_to_image_ids = return_images_context(image_ids)
    for i in text_to_image_ids:
        print(i)
        print(text_to_image_ids[i])
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

    # print(f"max_image content is: \n{max_image_context}")

    if max_image_context is None:
        return None, None

    image_emb = co.embed(texts=[max_image_context], model=model, input_type=input_type)
    val = float(cos_sim(query_emb.embeddings, image_emb.embeddings)[0][0])

    print(f"Image similarity value is {val}")
    if val >= img_threshold:
        return str(response.text), max_image_context

    else:
        return None, None
