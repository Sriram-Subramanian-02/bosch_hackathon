import base64
from transformers import CLIPProcessor, CLIPModel


def encode_image(image_path):
    """Getting the base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def load_clip_model(file_path="models"):
    model_ID = "openai/clip-vit-base-patch32"

    model = CLIPModel.from_pretrained(f"{file_path}/clip_model")
    processor = CLIPProcessor.from_pretrained(model_ID)

    return model, processor
