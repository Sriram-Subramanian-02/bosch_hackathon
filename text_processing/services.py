import time

import cohere
from sentence_transformers.util import cos_sim

from constants import COHERE_API_KEY_TABLES


def check_probing_conditions(context_list, query_emb, threshold):
    """
    Checks the similarity of a query embedding against a list of context embeddings.

    This function embeds the context list using a specified language model and then compares
    each embedding to the query embedding. It counts how many of the context embeddings
    have a similarity below the given threshold.

    Args:
        context_list (list of str): The list of context strings to be embedded and compared.
        query_emb: The embedding of the query for similarity comparison.
        threshold (float): The similarity threshold to consider for counting the embeddings.

    Returns:
        int: The count of context embeddings that have a similarity below the threshold.
    """

    co = cohere.Client(api_key=COHERE_API_KEY_TEXT)
    model = "embed-english-v3.0"
    input_type = "search_query"

    time.sleep(1)
    res = co.embed(texts=context_list, model=model, input_type=input_type)

    counter = 0
    for i in res.embeddings:
        if float(cos_sim(query_emb.embeddings, i)[0][0]) < threshold:
            print(float(cos_sim(query_emb.embeddings, i)[0][0]))
            counter += 1

    print(f"counter = {counter}")

    return counter
