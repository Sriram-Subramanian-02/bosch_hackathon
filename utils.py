import base64
from io import BytesIO
from pdf2image import convert_from_path


def get_pdf_pages(context):
    """
    Extracts the page numbers from the context of a PDF document.

    Args:
        context (list): A list of Document objects representing the context of a PDF document.

    Returns:
        dict: A dictionary where keys are car names and values are lists of page numbers.
    """

    pdf_pages = {}
    for doc in context:
        car_name = doc.metadata["car_name"]
        if car_name not in pdf_pages:
            pdf_pages[car_name] = [doc.metadata["page_number"]]
        else:
            pdf_pages[car_name].append(doc.metadata["page_number"])

    return pdf_pages


def pdf_to_images(pdf_path, pages=None):
    """
    Converts specified pages of a PDF document to base64-encoded PNG images.

    Args:
        pdf_path (str): The path to the PDF document.
        pages (list, optional): A list of page numbers to convert. Defaults to None, which converts all pages.

    Returns:
        list: A list of base64-encoded strings representing the PNG images.
    """

    images = []
    for page_num in pages:
        images.extend(
            convert_from_path(pdf_path, first_page=page_num + 1, last_page=page_num + 1)
        )

    image_paths = []
    for image in images:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        image_paths.append(img_str)
    return image_paths
