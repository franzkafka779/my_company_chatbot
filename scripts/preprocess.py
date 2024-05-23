from PyPDF2 import PdfFileReader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def pdf_to_text(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PdfFileReader(file)
        for page_num in range(reader.numPages):
            text += reader.getPage(page_num).extract_text()
    return text

def create_vector_db(text, db_path='db'):
    embedding = HuggingFaceEmbeddings(model_name='jhgan/ko-sroberta-multitask')
    vectordb = Chroma(persist_directory=db_path, embedding_function=embedding)
    vectordb.add_texts([text])
    return vectordb

if __name__ == "__main__":
    pdf_path = "../data/company_policy.pdf"
    text = pdf_to_text(pdf_path)
    vectordb = create_vector_db(text)
    vectordb.persist()
