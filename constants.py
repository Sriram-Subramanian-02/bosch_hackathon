import numpy as np


USER_ID = np.random.randint(10, 100000)
SESSION_ID = np.random.randint(10, 100000)
COHERE_API_KEY_TEXT = "YqDdRwGjdbF1N4ybPERe22CUOVzpahrVlqd3JBPs"
COHERE_API_KEY_IMAGES = "22XEk0Mtupvkeyh5rUeFQiYpGUkiRihpWajurshl"
COHERE_API_KEY_TABLES = "BhfDMsxb0C6RKV7KDBZEzMjWg9extEUwPyeX3cdC"

pdf_mapping = {
    "Hyundai Exter": "hyundai_exter.pdf",
    "Hyundai Verna": "Next_Gen_Verna.pdf",
    "Tata Nexon": "nexon-owner-manual-2022.pdf",
    "Tata Punch": "punch-bsvi-09-09-21.pdf",
}
