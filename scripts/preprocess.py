import os
from PyPDF2 import PdfReader, PdfWriter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def split_pdf(pdf_path, output_dir, chunk_size=10):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(pdf_path, "rb") as infile:
        reader = PdfReader(infile)
        total_pages = len(reader.pages)

        for start in range(0, total_pages, chunk_size):
            writer = PdfWriter()
            for i in range(start, min(start + chunk_size, total_pages)):
                writer.add_page(reader.pages[i])
            
            output_path = os.path.join(output_dir, f"split_{start // chunk_size}.pdf")
            with open(output_path, "wb") as outfile:
                writer.write(outfile)

def pdf_to_text(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PdfReader(file)
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
    return text

def create_vector_db_from_text(text, db_path):
    embedding = HuggingFaceEmbeddings(model_name='jhgan/ko-sroberta-multitask')
    vectordb = Chroma(persist_directory=db_path, embedding_function=embedding)
    vectordb.add_texts([text])
    return vectordb

def main():
    pdf_path = "./data/company_policy.pdf"
    split_dir = "./data/split_pdfs"
    db_dir = "./data/vector_dbs"

    # Step 1: Split PDF
    split_pdf(pdf_path, split_dir, chunk_size=10)

    # Step 2: Convert each split PDF to text and then to vector DB
    split_files = [os.path.join(split_dir, f) for f in os.listdir(split_dir) if f.endswith('.pdf')]
    for split_file in split_files:
        text = pdf_to_text(split_file)
        db_path = os.path.join(db_dir, os.path.basename(split_file).replace('.pdf', ''))
        vectordb = create_vector_db_from_text(text, db_path)
        vectordb.persist()

if __name__ == "__main__":
    main()
