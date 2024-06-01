import base64
from transformers import CLIPProcessor, CLIPModel


def encode_image(image_path):
    """
    Encodes an image file to a base64 string.

    Args:
        image_path (str): The file path to the image to be encoded.

    Returns:
        str: The base64 encoded string of the image.
    """

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def load_clip_model(file_path="models"):
    """
    Loads the CLIP model and processor from the specified directory.

    Args:
        file_path (str): The directory path where the CLIP model is stored. Default is "models".

    Returns:
        Tuple[CLIPModel, CLIPProcessor]: The loaded CLIP model and processor.
    """

    model_ID = "openai/clip-vit-base-patch32"

    model = CLIPModel.from_pretrained(f"{file_path}/clip_model")
    processor = CLIPProcessor.from_pretrained(model_ID)

    return model, processor
