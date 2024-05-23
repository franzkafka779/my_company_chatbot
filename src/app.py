import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
from googletrans import Translator
import os

st.header('Company Policy AI Chatbot', divider='rainbow')
st.markdown('''Feel free to ask anything about the company policies! :balloon:''')

# Load environment variables
load_dotenv()

# Initialize vector database and model
persist_directory = 'db'
embedding = HuggingFaceEmbeddings(model_name='jhgan/ko-sroberta-multitask')
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

class ChatGuide:
    def __init__(self, model_name='llama3'):
        self.model = ChatOllama(model=model_name)
        self.prompt = PromptTemplate.from_template(
            """
            You are an AI tutor answering questions about the company's policies.
            <Instructions>
            1. Provide accurate and detailed answers based on the given context.
            2. Use simple language suitable for the user's level while maintaining professional knowledge.
            3. Briefly explain key concepts or terms related to the answer.
            4. Use real-life examples or visual aids (images, graphs, diagrams) to help understand, encapsulated within <Example></Example> tags.
            5. Provide links or references to additional learning resources or documents if available.
            6. If it's challenging to answer based on the provided context, inform about the lack of information and request additional context.
            7. Answer in Korean.
            8. All responses must be in Korean only.
            </Instructions>
            <Context>
            {context}
            </Context>
            <User Question>
            {question}
            </User Question>
            <Answer>
            """
        )
        
        self.retriever = vectordb.as_retriever(search_kwargs={"k": 2})
        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.model  
            | StrOutputParser())
        
    def ask(self, query: str):
        return self.chain.invoke(query)

if "chat_guide" not in st.session_state:
    st.session_state["chat_guide"] = ChatGuide(model_name="llama3")

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
