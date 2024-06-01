import base64
from io import BytesIO
from pdf2image import convert_from_path


def get_pdf_pages(context):
    pdf_pages = {}
    for doc in context:
        car_name = doc.metadata["car_name"]
        if car_name not in pdf_pages:
            pdf_pages[car_name] = [doc.metadata["page_number"]]
        else:
            pdf_pages[car_name].append(doc.metadata["page_number"])

    return pdf_pages


def pdf_to_images(pdf_path, pages=None):
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
