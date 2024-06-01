from qdrant_client import models, QdrantClient

from databases.QDrant.constants import QDRANT_URL, QDRANT_API_KEY


def create_collection(collection_name):
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )

    client.create_collection(
        collection_name=f"{collection_name}",
        vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
    )


def delete_collection(collection_name):
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )

    client.delete_collection(collection_name=f"{collection_name}")


def upload_chunks_to_QDrant(qdrant_client, embedding_model, documents):
    records_to_upload = []
    for idx, chunk in enumerate(documents):
        content = chunk.page_content
        vector = embedding_model.encode(content).tolist()

        record = models.PointStruct(
            id=idx, vector=vector, payload={"page_content": content}
        )
        records_to_upload.append(record)

    qdrant_client.upload_points(
        collection_name="owners_manual", points=records_to_upload
    )
