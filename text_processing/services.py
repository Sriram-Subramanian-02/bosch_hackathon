import time

import cohere
from sentence_transformers.util import cos_sim

from constants import COHERE_API_KEY_TEXT


def check_probing_conditions(context_list, query_emb, threshold):
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
