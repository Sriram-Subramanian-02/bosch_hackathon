import os
from itertools import chain
from operator import itemgetter

from langchain.load import dumps, loads
from langchain_core.prompts import ChatPromptTemplate
from langchain.llms import Cohere
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents.base import Document
from langchain.embeddings.cohere import CohereEmbeddings
from qdrant_client import models, QdrantClient
from langchain.vectorstores import Qdrant
from langchain_core.runnables import RunnableLambda

from constants import COHERE_API_KEY_TABLES
from retrieval_augmented_generation.constants import MAX_DOCS_FOR_CONTEXT, TOP_K
from databases.QDrant.constants import (
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
)

os.environ["COHERE_API_KEY"] = COHERE_API_KEY_TABLES


def reciprocal_rank_fusion(results: list[list], k=60):
    """Rerank docs (Reciprocal Rank Fusion)

    Args:
        results (list[list]): retrieved documents
        k (int, optional): parameter k for RRF. Defaults to 60.

    Returns:
        ranked_results: list of documents reranked by RRF
    """

    # print("\n\n\nresults in rrf function: ", len(results))

    fused_scores = {}
    for docs in results:
        # Assumes the docs are returned in sorted order of relevance
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # return only documents
    return [x[0] for x in reranked_results[:MAX_DOCS_FOR_CONTEXT]]


def query_generator(original_query: dict) -> list[str]:
    """Generate queries from original query

    Args:
        query (dict): original query

    Returns:
        list[str]: list of generated queries
    """

    # original query
    query = original_query.get("query")

    # prompt for query generator
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that generates multiple search queries based on a single input query.",
            ),
            (
                "user",
                "Generate multiple search queries related to:  {original_query}. When creating queries, please refine or add closely related contextual information, without significantly altering the original query's meaning",
            ),
            ("user", "OUTPUT (3 queries):"),
        ]
    )

    # LLM model
    model = Cohere()

    # query generator chain
    query_generator_chain = (
        prompt | model | StrOutputParser() | (lambda x: x.split("\n"))
    )

    # gererate queries
    queries = query_generator_chain.invoke({"original_query": query})

    # add original query
    queries.insert(0, "0. " + query)

    return queries


def rrf_retriever(query: str) -> list[Document]:
    """RRF retriever

    Args:
        query (str): Query string

    Returns:
        list[Document]: retrieved documents
    """

    # Retriever
    embedding = CohereEmbeddings(model="embed-english-v3.0")

    qdrant_client = QdrantClient(
        QDRANT_URL,
        prefer_grpc=True,
        api_key=QDRANT_API_KEY,
    )

    qdrant = Qdrant(
        client=qdrant_client,
        collection_name=QDRANT_COLLECTION_NAME,
        embeddings=embedding,
    )

    retriever = qdrant.as_retriever(
        search_kwargs={"k": TOP_K}, metadata={"car_name": "Hyundai Exter"}
    )

    # RRF chain
    chain = (
        {"query": itemgetter("query")}
        | RunnableLambda(query_generator)
        | retriever.map()
        | reciprocal_rank_fusion
    )

    # invoke
    result = chain.invoke({"query": query})

    image_ids = []
    from itertools import chain

    for document in result:
        image_ids.append(document.metadata["image_ids"])
    image_ids = list(
        chain.from_iterable(
            [item] if isinstance(item, str) else item
            for item in image_ids
            if item is not None
        )
    )

    print(image_ids)
    return result, image_ids


def normal_retriever(query: str) -> list[Document]:
    """
    Retrieves relevant documents and associated metadata based on the input query.

    This function uses the Cohere embedding model to embed the query and the Qdrant client
    to retrieve relevant documents from a specified collection. It processes the results to
    extract image IDs and table data.

    Args:
        query (str): The input query string.

    Returns:
        Tuple[List[Document], List[str], List[str]]: A tuple containing:
            - List[Document]: The list of retrieved documents.
            - List[str]: The list of image IDs extracted from the metadata of the documents.
            - List[str]: The list of table data extracted from the content of the documents.
    """

    embedding = CohereEmbeddings(model="embed-english-v3.0")

    qdrant_client = QdrantClient(
        QDRANT_URL,
        prefer_grpc=True,
        api_key=QDRANT_API_KEY,
    )
    print(qdrant_client)

    qdrant = Qdrant(
        client=qdrant_client,
        collection_name=QDRANT_COLLECTION_NAME,
        embeddings=embedding,
    )

    retriever = qdrant.as_retriever(search_kwargs={"k": TOP_K}, metadata={})

    # invoke
    result = retriever.invoke(query)
    print(result)

    image_ids = []
    table_data = list()

    for document in result:
        image_ids.append(document.metadata["image_ids"])
        if document.metadata["chunk_type"] == "Table":
            table_data.append(document.page_content)
    image_ids = list(
        chain.from_iterable(
            [item] if isinstance(item, str) else item
            for item in image_ids
            if item is not None
        )
    )

    print(image_ids)

    # if len(table_data) > 0:
    #     table_data = str(table_data[0])
    #     return result, image_ids, table_data

    # else:
    return result, image_ids, table_data
