import os
import time
from PIL import Image
import requests

import torch
from qdrant_client import QdrantClient

from databases.QDrant.constants import (
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_IMAGE_COLLECTION_NAME,
)
from image_processing.utils import encode_image, load_clip_model
from image_processing.services import return_images_context
from image_processing.constants import ROBOFLOW_API_KEY


def get_image_context_from_QDrant(image_vector):
    """
    Retrieves the context for the image closest to the given image vector from Qdrant.

    This function searches the Qdrant collection for the image most similar to the provided
    image vector. It retrieves the context of the closest image using its ID and returns the
    summary of the image context.

    Args:
        image_vector (List[float]): The vector representation of the image to search for.

    Returns:
        str: The summary of the context of the most similar image found.
    """

    qdrant_client = QdrantClient(
        QDRANT_URL,
        prefer_grpc=True,
        api_key=QDRANT_API_KEY,
    )

    search_result = qdrant_client.search(
        collection_name=QDRANT_IMAGE_COLLECTION_NAME,
        query_vector=image_vector,
        limit=1,
    )

    payloads = [hit.payload for hit in search_result]
    print(payloads)
    image_id = payloads[0]["metadata"]["image_id"]
    image_context = return_images_context([image_id])
    image_summary = list(image_context.keys())[0]
    print(image_summary)
    return image_summary


def get_image_summary_roboflow(image_path):
    """
    Retrieves a summary of the context for an image using the Roboflow API and Qdrant.

    This function encodes an image, sends it to the Roboflow API to get its embeddings,
    and then searches for the most similar image in the Qdrant collection to retrieve
    its context summary.

    Args:
        image_path (str): The file path of the image to be processed.

    Returns:
        str: The summary of the context for the most similar image found.
        Returns None if embeddings are not found in the response.
    """

    encoded_val = encode_image(image_path)
    infer_clip_payload = {
        "image": {
            "type": "base64",
            "value": f"{encoded_val}",
        },
    }
    base_url = "https://infer.roboflow.com"

    res = requests.post(
        f"{base_url}/clip/embed_image?api_key={ROBOFLOW_API_KEY}",
        json=infer_clip_payload,
    )

    embeddings = res.json()

    if "embeddings" in embeddings:
        return get_image_context_from_QDrant(embeddings["embeddings"][0])
    else:
        return None


def get_image_summary_clip(image_bytes, image_format):
    """
    Retrieves a summary of the context for an image using the CLIP model and Qdrant.

    This function loads the CLIP model, processes the input image to get its embeddings,
    and then searches for the most similar image in the Qdrant collection to retrieve
    its context summary.

    Args:
        image_bytes (bytes): The byte content of the image to be processed.
        image_format (str): The format of the image (e.g., 'png', 'jpeg').

    Returns:
        str: The summary of the context for the most similar image found.
    """

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
    with open(image_path, "wb") as f:
        f.write(image_bytes)

    my_image = Image.open(image_path)

    image = processor(text=None, images=my_image, return_tensors="pt")[
        "pixel_values"
    ].to(device)

    # Get the image features
    embedding = model.get_image_features(image)
    embedding_as_np = embedding.cpu().detach().numpy()
    return get_image_context_from_QDrant(embedding_as_np)
