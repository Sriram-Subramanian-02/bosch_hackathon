import numpy as np


USER_ID = np.random.randint(10, 100000)
SESSION_ID = np.random.randint(10, 100000)
COHERE_API_KEY_1 = "8y2cASrkSAGav1wi6VTlubFBMp7V1XRhMsy72leR"
COHERE_API_KEY_2 = "xsVMq3dnuekhW5miz2Cq0HLXAMGnAjXwuSM9PDtk"
COHERE_API_KEY = "P6t9MllaPbgvUFgMVhA3VGnrSYfAE3AP1PDdZdV9"
QDRANT_URL = "https://8803fa99-7551-4f88-84c3-e134c9bed5de.us-east4-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "EFeN_UhdmAlDNYZHqJBUbZ88Nt7N0MkmvWLgM5Hs4ogNvExLMwNwdQ"
QDRANT_COLLECTION_NAME = "owners_manual_chunks"
MONGO_DB_URL = "mongodb+srv://sriram:Ayynar%40123@msd.ywfrjgy.mongodb.net/?retryWrites=true&w=majority"

pdf_mapping = {
                "Hyundai Exter": "hyundai_exter.pdf",
                "Hyundai Verna": "Next_Gen_Verna.pdf",
                "Tata Nexon": "nexon-owner-manual-2022.pdf",
                "Tata Punch": "punch-bsvi-09-09-21.pdf"
                }
            