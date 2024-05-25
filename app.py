import streamlit as st
import base64
import time

from utils import insert_data, get_full_data, get_file_details
from constants import USER_ID, SESSION_ID, MONGO_DB_URL, QDRANT_API_KEY, QDRANT_URL, COHERE_API_KEY
from services import get_response

st.set_page_config(page_title="BOSCH Hackathon Chatbot")
st.title("BOSCH Hackathon Chatbot")


user_question = st.chat_input("What is up?")
# prev_records = check_and_delete_existing_records(USER_ID, SESSION_ID)
# print(prev_records)
full_data = get_full_data(USER_ID, SESSION_ID)
full_data.reverse()

for message in full_data:
    with st.chat_message("user"):
        st.markdown(message["query"])
    with st.chat_message("assistant"):
        st.markdown(message["response"])


if user_question:
    start_time = time.time()
    response, image_id = get_response(user_question)
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"\n\n\nExecution time: {execution_time} seconds")

    print(image_id)

    insert_data(USER_ID, SESSION_ID, user_question, response)

    with st.chat_message("user"):
        st.write(f"{user_question}")

    with st.chat_message("assistant"):
        st.write(f"{response}")

        # Display the image if image_id is provided and valid
        if image_id:
            try:
                # Decode the base64-encoded image
                img_bytes = base64.b64decode(get_file_details(image_id)['encoded_val'])
                st.image(img_bytes, caption="Related Image", use_column_width=True)
            except Exception as e:
                st.error(f"Failed to display image: {e}")


