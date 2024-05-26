# LLM RAG Chat-Q&A Application
This repository contains code for a question-and-answer (Q&A) application powered by a Large Language Model (LLM) with Retrieval-Augmented Generation (RAG) techniques.

## Project Description
This application utilizes an LLM to process user queries and provide informative answers. It leverages RAG techniques to:

Retrieve relevant textual content (including tables) from a Vector Database.
Generate responses that are both accurate and relevant to the user's intent.
Employ probing questions to gain further context and refine understanding of the user's prompt.
The application strives to deliver responses within a minute of receiving a user query.

## Features
Access and process textual data and tables.
Leverage LLM capabilities for response generation.
Utilize RAG techniques for enhanced retrieval and response accuracy.
Incorporate probing questions for improved user query clarity.
Aim for sub-minute response latency.
Includes a basic user interface for a chat-style Q&A experience.

## Dependencies
This project relies on specific libraries and frameworks for LLM access, Vector Database interaction, and UI development. Refer to the requirements.txt file for a detailed list.

## Installation: 
Clone this repository:
```
git clone https://github.com/Sriram-Subramanian-02/bosch_hackathon
```
## Install required dependencies:
```
pip install -r requirements.txt
```
## Run the application locally
```
streamlit run app.py
```

## License
This project is licensed under the [License Name] license. See the LICENSE file for details.

## Deployed URL
https://boschhackathon.streamlit.app/