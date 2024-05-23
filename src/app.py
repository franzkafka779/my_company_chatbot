import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from googletrans import Translator
from dotenv import load_dotenv
import os

#__import__('pysqlite3')
import sqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('sqlite3')

st.header('Company Policy AI Chatbot', divider='rainbow')
st.markdown('''Feel free to ask anything about the company policies! :balloon:''')

# 환경 변수 로드
load_dotenv()

# 여러 벡터 데이터베이스를 로드하는 함수
def load_vector_dbs(db_dir):
    embedding = HuggingFaceEmbeddings(model_name='jhgan/ko-sroberta-multitask')
    vector_dbs = []
    for db_path in os.listdir(db_dir):
        full_path = os.path.join(db_dir, db_path)
        if os.path.isdir(full_path):
            vectordb = Chroma(persist_directory=full_path, embedding_function=embedding)
            vector_dbs.append(vectordb)
    return vector_dbs

# Vector DBs 로드
base_dir = os.path.abspath(os.path.dirname(__file__))
db_dir = os.path.join(base_dir, '..', 'db')  # 실제 벡터 데이터베이스 폴더 경로로 변경하세요
vector_dbs = load_vector_dbs(db_dir)

# 여러 벡터 데이터베이스에서 검색할 수 있도록 설정
class MultiVectorRetriever:
    def __init__(self, vector_dbs):
        self.vector_dbs = vector_dbs
    
    def retrieve(self, query):
        results = []
        for vectordb in self.vector_dbs:
            # results.extend(vectordb.as_retriever(search_kwargs={"k": 2}).retrieve(query))
            results.extend(vectordb.similarity_search(query, k=2))
        return results

retriever = MultiVectorRetriever(vector_dbs)

class ChatGuide:
    def __init__(self, retriever, model_name='llama3'):
        self.model = ChatOllama(model=model_name)
        self.prompt = PromptTemplate.from_template(
            """
            당신은 회사 상품정책에 대한 질문에 답변하는 AI 챗봇입니다.
            <지침>
            1. 제공된 컨텍스트를 바탕으로 질문에 대해 최대한 정확하고 상세하게 답변해 주세요.
            2. 답변은 질문자가 이해하기 쉬운 언어를 사용하되, 전문적인 지식을 바탕으로 작성해 주세요.
            3. 답변에 관련된 핵심 개념이나 용어가 있다면 간단히 설명해 주세요.
            4. 실생활 예시나 시각 자료(이미지, 그래프, 다이어그램 등)를 활용하여 이해를 돕는 것이 좋습니다. 예시는 <예시></예시> 태그로 감싸 주세요.
            5. 추가 학습에 도움이 될 만한 자료나 참고 문헌이 있다면 링크 또는 출처를 제공해 주세요.
            6. 제공된 컨텍스트만으로 답변하기 어려운 경우, 관련 정보가 부족함을 알리고 추가 컨텍스트를 요청해 주세요.
            7. 반드시 답변은 한국어로 작성해 주세요. 다른 언어를 사용하지 마세요.
            </지침>
            <컨텍스트>
            {context}
            </컨텍스트>
            <질문>
            {question}
            </질문>
            <답변>
            """
        )
        
        self.retriever = retriever
        self.chain = ({"context": self.retriever.retrieve, "question": RunnablePassthrough()}
            | self.prompt
            | self.model  
            | StrOutputParser())
        
    def ask(self, query: str):
        return self.chain.invoke(query)

# ChatGuide 초기화
if "chat_guide" not in st.session_state:
    st.session_state["chat_guide"] = ChatGuide(retriever=retriever, model_name="llama3")

# 번역기 초기화
translator = Translator()

# Streamlit 애플리케이션 구성
st.title("포커스미디어 Chatbot")

with st.chat_message("user"):
    st.write("부산지하철 상품구좌사이클에 관해 알려주세요.")
    st.write("브랜드정보 제작투자비 쿠폰 제공 조건을 알려주세요.")
    st.write("문화정보 상품의 서울생활권 금액을 알려주세요.")
                 
# 대화 기록 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []

def model_res_generator(query):
    response = st.session_state["chat_guide"].ask(query)
    return response

# 대화 기록 표시
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    # 사용자 입력 기록
    st.session_state["messages"].append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 모델 응답 생성
    response = model_res_generator(prompt)
    
    # 모델 응답을 한국어로 번역
    translated_response = translator.translate(response, src='en', dest='ko').text
    
    with st.chat_message("assistant"):
        st.markdown(translated_response)
        
    st.session_state["messages"].append({"role": "assistant", "content": translated_response})
