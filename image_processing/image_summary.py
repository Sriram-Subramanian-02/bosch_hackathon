import os
import time
from PIL import Image

import torch
from qdrant_client import QdrantClient

from image_processing.utils import encode_image, load_clip_model
from image_processing.services import return_images_context


def get_image_context_from_QDrant(image_vector):
    qdrant_client = QdrantClient(
        "https://35ebdc7d-ec99-4ebd-896c-ff5705cf369b.us-east4-0.gcp.cloud.qdrant.io:6333",
        prefer_grpc=True,
        api_key="9dKJsKOYwT0vGlWPrZXBSIlbUzvRdJ1XkM0_floo8FmYCOHX_Y0y-Q",
    )

    search_result = qdrant_client.search(
        collection_name="owners_manual_images_roboflow",
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
    import requests

    encoded_val = encode_image(image_path)
    infer_clip_payload = {
        "image": {
            "type": "base64",
            "value": f"{encoded_val}",
        },
    }
    base_url = "https://infer.roboflow.com"
    api_key = "pUmnI6Vv3mdDdmDiEtqz"

    res = requests.post(
        f"{base_url}/clip/embed_image?api_key={api_key}",
        json=infer_clip_payload,
    )

    embeddings = res.json()

    if "embeddings" in embeddings:
        return get_image_context_from_QDrant(embeddings["embeddings"][0])
    else:
        return None


def get_image_summary_clip(image_bytes, image_format):
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
