import json
import pandas as pd

import cohere
from sentence_transformers.util import cos_sim

from constants import COHERE_API_KEY_TABLES


def reconstruct_table(table_data, context, query, query_emb, table_threshold=0.5):
    model = "embed-english-v3.0"
    input_type = "search_query"

    prompt = f"""
        Reconstruct the table using this table data: {table_data}, question: {query} and context: {context}.
        Reconstruct only one element of table_data that is most similar to the question.
        Return the table in json format and do not add anything else like new line characters or tab spaces.
        If car names in question and table_data does not match return empty string.
        If you feel that json format that you are returning is not similar to question, return empty string.
    """

    co = cohere.Client(COHERE_API_KEY_TABLES)
    response = co.chat(message=prompt, model="command-r", temperature=0)
    response = response.text
    print(response)
    if response == "" or response is None or response == "None":
        return None, None
    formatted_response = (
        response.replace("\n", "")
        .replace("\t", "")
        .replace("`", "")
        .replace("json", "")
    )
    print(formatted_response)
    try:
        data = json.loads(formatted_response)
        json_string = json.dumps(data)

        table_emb = co.embed(texts=[json_string], model=model, input_type=input_type)
        val = float(cos_sim(query_emb.embeddings, table_emb.embeddings)[0][0])
        print(val)
        if val < table_threshold:
            return None, None

        print(json.dumps(data, indent=4))
        try:
            df = pd.DataFrame(data)

            return df, formatted_response

        except:
            return None, data

    except:
        return None, formatted_response
