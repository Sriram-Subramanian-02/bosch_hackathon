import streamlit as st
from utils import insert_data, get_full_data
from constants import USER_ID, SESSION_ID, MONGO_DB_URL, QDRANT_API_KEY, QDRANT_URL, COHERE_API_KEY
from services import get_response

st.set_page_config(page_title="BOSCH Hackathon Chatbot")
st.title("BOSCH Hackathon Chatbot")


user_question = st.chat_input("What is up?")
full_data = get_full_data(USER_ID, SESSION_ID)
full_data.reverse()

for message in full_data:
    with st.chat_message("user"):
        st.markdown(message["query"])
    with st.chat_message("assistant"):
        st.markdown(message["response"])


if user_question:
    response = get_response(user_question)

    insert_data(USER_ID, SESSION_ID, user_question, response)

    with st.chat_message("user"):
        st.write(f"{user_question}")

    with st.chat_message("assistant"):
        st.write(f"{response}")


