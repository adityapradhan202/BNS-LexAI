from dotenv import load_dotenv
import os
from langchain_pinecone import PineconeEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
import re

load_dotenv()
pinecone_key = os.getenv("PINECONE_KEY")
pdf_path = os.getenv("PDF_PATH")

embedding_model = PineconeEmbeddings(model='llama-text-embed-v2', pinecone_api_key=pinecone_key)
pc = Pinecone(api_key=pinecone_key)
index = pc.Index('bns-lex-ai')

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def create_vector_database(index=index, embedding_model=embedding_model, pdf_path=pdf_path):
    print("\n-> Initializing vectore store creation")
    print("-> Creating chunks")
    loader = PyPDFLoader(file_path=pdf_path, mode='page')
    pages = loader.load()
    for page in pages:
        page.page_content = clean_text(page.page_content)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)
    print(f"-> Successfully created chunks. Total chunks: {len(chunks)}")

    vector_store = PineconeVectorStore(index=index, embedding=embedding_model)
    try:
        print("-> Adding chunks to vectore store")
        vector_store.add_documents(
            documents=chunks
        )
        print("-> Successfully added chunks to the vectore store")
    except Exception as e:
        print(f"Exception occured: {e}")

if __name__ == "__main__":
    print("\n[DANGER ZONE] ⚠️")
    print("Are you sure you want to add documents?")
    print("Check if the index already exists on pinecone-console.")
    print("Make sure that you are not adding the same chunks again, you will waste your write compute units.")
    x = input("Enter Y to proceed, N to cancel: ")
    if x.lower() == "y":
        create_vector_database(index=index, embedding_model=embedding_model, pdf_path=pdf_path)
    elif x.lower() == "n":
        print("Process cancelled!")
    else:
        print("Invalid input!")

