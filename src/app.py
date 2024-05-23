import os
import streamlit as st
import threading
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
from googletrans import Translator
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import requests
from llama_server import run_server

# Start Llama server in a separate thread
server_thread = threading.Thread(target=run_server)
server_thread.daemon = True
server_thread.start()

st.header('Company Policy AI Chatbot', divider='rainbow')
st.markdown('''Feel free to ask anything about the company policies! :balloon:''')

# Load environment variables
load_dotenv()

# Llama 모델 서버 URL
llama_server_url = os.getenv('LLAMA_SERVER_URL', 'http://localhost:5000/generate')

class ChatGuide:
    def __init__(self, server_url):
        self.server_url = server_url
        
    def ask(self, query: str):
        response = requests.post(self.server_url, json={"input_text": query})
        return response.json().get("response", "")

if "chat_guide" not in st.session_state:
    st.session_state["chat_guide"] = ChatGuide(server_url=llama_server_url)

translator = Translator()

st.title("Company Policy Chatbot")

# Display chat messages
if "messages" not in st.session_state:
    st.session_state["messages"] = []

def model_res_generator(query):
    response = st.session_state["chat_guide"].ask(query)
    return response

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    response = model_res_generator(prompt)
    translated_response = translator.translate(response, src='en', dest='ko').text
    
    with st.chat_message("assistant"):
        st.markdown(translated_response)
        
    st.session_state["messages"].append({"role": "assistant", "content": translated_response})
