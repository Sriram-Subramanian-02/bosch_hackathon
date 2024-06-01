import json
import pandas as pd

import cohere
from sentence_transformers.util import cos_sim

from constants import COHERE_API_KEY_TABLES


def reconstruct_table(table_data, context, query, query_emb, table_threshold=0.5):
    """
    Reconstructs a table based on provided data, context, and query.

    This function uses a language model to generate a JSON representation of a table element
    that is most relevant to the given query. The reconstructed table is validated for similarity
    to the query and returned if it meets the similarity threshold.

    Args:
        table_data (str): The data of the table to be reconstructed.
        context (str): Additional context for the reconstruction process.
        query (str): The query to guide the reconstruction.
        query_emb: The embedding of the query for similarity comparison.
        table_threshold (float): The similarity threshold to validate the reconstructed table. Default is 0.5.

    Returns:
        tuple: A tuple containing the DataFrame of the reconstructed table and the JSON string of the table.
               Returns (None, None) if the reconstruction is not valid or doesn't meet the similarity threshold.
    """

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
