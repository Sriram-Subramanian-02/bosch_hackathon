import streamlit as st
import base64
import time

from file_chat_input import file_chat_input
from streamlit_float import float_init

from utils import insert_data, get_full_data, get_file_details
from constants import USER_ID, SESSION_ID, pdf_mapping
from services import get_response, pdf_to_images, get_single_image_embedding 

float_init()

def page_switcher(page):
    st.session_state.runpage = page

def main():
    st.title("BOSCH Hackathon Chatbot")
    container = st.container()
    with container:
        user_input = file_chat_input("What is up?")
    
    # prev_records = check_and_delete_existing_records(USER_ID, SESSION_ID)
    # print(prev_records)
    full_data = get_full_data(USER_ID, SESSION_ID)
    full_data.reverse()

    for message in full_data:
        with st.chat_message("user"):
            st.markdown(message["query"])
        with st.chat_message("assistant"):
            st.markdown(message["response"])


    if user_input:

        if len(user_input['files']) == 0:
            start_time = time.time()
            response, image_id, pdf_pages = get_response(user_input['message'])
            end_time = time.time()

            execution_time = end_time - start_time
            print(f"\n\n\nExecution time: {execution_time} seconds")

            print(image_id)

            insert_data(USER_ID, SESSION_ID, user_input['message'], response)

            with st.chat_message("user"):
                st.write(f"{user_input['message']}")

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
                    st.session_state.show_pdf_btn = True  # Set the flag to show the button

        if user_input['message'] == '' and len(user_input['files']) != 0:
            base64_string = user_input['files'][0]['content']
            if base64_string.startswith('data:image/png;base64,'):
                base64_string = base64_string.replace('data:image/png;base64,', '')

            image_data = base64.b64decode(base64_string)
            image_embedding = get_single_image_embedding(image_data)
            print("..........",len(image_embedding))

            with st.chat_message("user"):
                st.image(image_data, caption="Uploaded Image", use_column_width=True)
            with st.chat_message("assistant"):
                st.write(f"hiiii")


    if st.session_state.get("show_pdf_btn", False):  # Check the flag
        if st.button('View Reference PDF Contents'):
            page_switcher(reference_pdf)
            st.rerun()

    container.float("bottom: 0")

def reference_pdf():
    st.title('Reference PDF')
    pdf_pages = st.session_state.pdf_pages
    if st.button('Go back to Chat'):
        st.session_state.runpage = main        
        st.rerun()

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

if __name__ == '__main__':
    if 'runpage' not in st.session_state:
        st.session_state.runpage = main
    st.session_state.runpage()
