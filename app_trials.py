import streamlit as st
import base64
import time
import shutil
import os

from constants import USER_ID, SESSION_ID, pdf_mapping
from databases.MongoDB.utils import insert_data, get_full_data, get_file_details
from image_processing.image_summary import get_image_summary_roboflow
from services import get_response
from utils import pdf_to_images


file_names = dict()



def page_switcher(page):
    st.session_state.runpage = page

def main():
    st.set_page_config(page_title="BOSCH Hackathon Chatbot")
    st.title("BOSCH Hackathon Chatbot")


    user_question = st.chat_input("Ask a Question...")
    file_exists = False
    
    uploaded_file = st.file_uploader(label="Choose a file", type=['png', 'jpg', 'jpeg'], accept_multiple_files=False)
    # if uploaded_file:
    #     st.write("filename:", uploaded_file.name)
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
        response, image_id, pdf_pages, df, table_response = None, None, None, None, None
        if (not uploaded_file) or (uploaded_file and uploaded_file.name in list(file_names.keys())):
            print("\n--------------------Question with Text--------------------\n")
            start_time = time.time()
            response, image_id, pdf_pages, df, table_response = get_response(user_question)
            end_time = time.time()

            execution_time = end_time - start_time
            print(f"\n\n\nExecution time: {execution_time} seconds")

            print(image_id)

            insert_data(USER_ID, SESSION_ID, user_question, response)

            with st.chat_message("user"):
                st.write(f"{user_question}")

            with st.chat_message("assistant"):
                st.write(f"{response}")

                if df is not None:
                    st.dataframe(df)
                    st.write("Related Table")
                    st.write(f"{table_response}")
                
                if df is None and table_response is not None:
                    st.write(f"{table_response}")
                    st.write("Related JSON Data")

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

        elif uploaded_file and uploaded_file.name not in list(file_names.keys()):
            print("\n--------------------Question with Image and Text--------------------\n")
            print(f"\nfile_names = {file_names}")
            print(f"\nuploaded file name is {uploaded_file.name}")
            file_names[uploaded_file.name] = time.time()
            print(f"\nfile_names = {file_names}")

            print("\n\n\n")
            # image_bytes = uploaded_file.read()
            # image_format = uploaded_file.type.split('/')[1]  # Get the image format from the MIME type
            # image_summary = get_image_summary(image_bytes, image_format)
            image_bytes = uploaded_file.read()
            image_format = uploaded_file.type.split('/')[1]  # Get the image format from the MIME type

            # Save the uploaded image to a local path
            input_image_directory_path = "input_data/user_image_input"
            if not os.path.exists(input_image_directory_path):
                os.makedirs(input_image_directory_path, exist_ok=True)
            
            image_path = f"{input_image_directory_path}/input_image.{image_format}"
            
            # Check if the file already exists and delete it
            if os.path.exists(image_path):
                file_exists = True
                os.remove(image_path)
            else:
                file_exists = False
                
            
            # Write the new image file
            with open(image_path, 'wb') as f:
                f.write(image_bytes)

            image_summary = get_image_summary_roboflow(image_path)

            query = f"""{image_summary} - This is a summary of an image uploaded by the user, 
            with this data answer the following question {user_question}"""

            response, image_id, pdf_pages, df, table_response = get_response(query)

            with st.chat_message("user"):
                st.image(image_bytes, caption="Uploaded Image", use_column_width=True)
                st.write(f"{user_question}")
            
            input_image_directory_path = "input_data/user_image_input"
            if os.path.exists(input_image_directory_path) and os.path.isdir(input_image_directory_path):
                shutil.rmtree(input_image_directory_path)

            insert_data(USER_ID, SESSION_ID, user_question, response)

            with st.chat_message("assistant"):
                st.write(f"{response}")

                if df is not None:
                    st.dataframe(df)
                    st.write("Related Table")
                    st.write(f"{table_response}")
                
                if df is None and table_response is not None:
                    st.write(f"{table_response}")
                    st.write("Related JSON Data")

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


    if st.session_state.get("show_pdf_btn", False):  # Check the flag
        if st.button('View Reference PDF Contents'):
            page_switcher(reference_pdf)
            st.rerun()


def reference_pdf():
    st.title("Reference PDF")
    pdf_pages = st.session_state.pdf_pages
    if st.button("Go back to Chat"):
        st.session_state.runpage = main
        st.rerun()

    carousel_indicators = ""
    carousel_items = ""
    page_num = 0

    for car_name in pdf_pages.keys():
        image_paths = pdf_to_images(
            f"./input_data/{pdf_mapping[car_name]}", list(set(pdf_pages[car_name]))
        )
        for i, img_str in enumerate(image_paths):
            active_class = "active" if page_num == 0 else ""
            carousel_indicators += f'<li data-target="#myCarousel" data-slide-to="{page_num}" class="{active_class}"></li>'
            carousel_items += f"""
            <div class="item {active_class}">
                <img src="data:image/png;base64,{img_str}" alt="Page {page_num+1}" style="width:100%;">
            </div>
            """
            page_num += 1

    path_to_html = "./display_carousel.html"
    with open(path_to_html, "r") as f:
        html_data = f.read()

    html_data = html_data.replace("{{carousel_indicators}}", carousel_indicators)
    html_data = html_data.replace("{{carousel_items}}", carousel_items)

    st.components.v1.html(html_data, height=1000)


if __name__ == "__main__":
    if "runpage" not in st.session_state:
        st.session_state.runpage = main
    st.session_state.runpage()
