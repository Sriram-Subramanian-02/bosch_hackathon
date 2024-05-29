import streamlit as st
import base64
import time

from utils import insert_data, get_full_data, get_file_details
from constants import USER_ID, SESSION_ID, pdf_mapping
from services import get_response, pdf_to_images


def page_switcher(page):
    st.session_state.runpage = page

def main():
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
        response, image_id, pdf_pages = get_response(user_question)
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

            if pdf_pages:
                st.session_state.pdf_pages = pdf_pages
                show_pdf_btn = st.button('View Reference PDF Contents',on_click=page_switcher,args=(reference_pdf,))
                if show_pdf_btn:
                    st.rerun() 

def reference_pdf():
    st.title('Reference PDF')
    pdf_pages = st.session_state.pdf_pages
    back_to_chat_btn = st.button('Go back to Chat')

    carousel_indicators = ""
    carousel_items = ""
    page_num = 0

    for car_name in pdf_pages.keys():
        image_paths = pdf_to_images(f"./input_data/{pdf_mapping[car_name]}", list(set(pdf_pages[car_name])))
        for i, img_str in enumerate(image_paths):
            active_class = "active" if page_num == 0 else ""
            carousel_indicators += f'<li data-target="#myCarousel" data-slide-to="{page_num}" class="{active_class}"></li>'
            carousel_items += f'''
            <div class="item {active_class}">
                <img src="data:image/png;base64,{img_str}" alt="Page {page_num+1}" style="width:100%;">
            </div>
            '''
            page_num += 1

    path_to_html = "./display_carousel.html"
    with open(path_to_html, 'r') as f:
        html_data = f.read()

    html_data = html_data.replace("{{carousel_indicators}}", carousel_indicators)
    html_data = html_data.replace("{{carousel_items}}", carousel_items)

    st.components.v1.html(html_data, height=1000)

    if back_to_chat_btn :
        st.session_state.runpage = main        
        st.rerun()

if __name__ == '__main__':
    if 'runpage' not in st.session_state :
        st.session_state.runpage = main
    st.session_state.runpage()
