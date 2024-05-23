from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain.chains.question_answering import load_qa_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os

class ChatGuide:
    def __init__(self, model_name='llama3', persist_directory='db'):
        self.model = ChatOllama(model=model_name)
        self.prompt = PromptTemplate.from_template(
            """
            당신은 회사 내규에 대해 질문하는 사용자에게 답변하는 AI 튜터입니다.
            <지침>
            1. 제공된 컨텍스트를 바탕으로 사용자 질문에 대해 최대한 정확하고 상세하게 답변해 주세요.
            2. 답변은 사용자의 수준에 맞는 쉬운 언어를 사용하되, 전문적인 지식을 바탕으로 작성해 주세요.
            3. 답변에 관련된 핵심 개념이나 용어가 있다면 간단히 설명해 주세요.
            4. 실생활 예시나 시각 자료(이미지, 그래프, 다이어그램 등)를 활용하여 이해를 돕는 것이 좋습니다. 예시는 <예시></예시> 태그로 감싸 주세요.
            5. 추가 학습에 도움이 될 만한 자료나 참고 문헌이 있다면 링크 또는 출처를 제공해 주세요.
            6. 제공된 컨텍스트만으로 답변하기 어려운 경우, 관련 정보가 부족함을 알리고 추가 컨텍스트를 요청해 주세요.
            7. 반드시 답변은 한국어로만 해주세요.
            8. 모든 답변은 한국어로 작성해 주세요. 다른 언어를 사용하지 마세요.
            </지침>
            <컨텍스트>
            {context}
            </컨텍스트>
            <사용자 질문>
            {question}
            </사용자 질문>
            <답변>
            """
        )
        embedding = HuggingFaceEmbeddings(model_name='jhgan/ko-sroberta-multitask')
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
        self.retriever = vectordb.as_retriever(search_kwargs={"k": 2})
        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                      | self.prompt
                      | self.model
                      | StrOutputParser())
        
    def ask(self, query: str):
        return self.chain.invoke(query)
