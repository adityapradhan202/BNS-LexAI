from dotenv import load_dotenv
import os
from langchain_pinecone import PineconeEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()
pinecone_key = os.getenv("PINECONE_KEY")

embedding_model = PineconeEmbeddings(model='llama-text-embed-v2', pinecone_api_key=pinecone_key)
pc = Pinecone(api_key=pinecone_key)
index = pc.Index('bns-lex-ai')

async def fetch_docs(query:str, index=index, embedding_model=embedding_model) -> str:
    """Retrieves top 3 documents from pinecone vectore store. It uses similarity search.
    Args:
        query(str): user query
        index: pinecone's index object
        embedding_function: pinecone's embedding function
    Returns:
        context(str): A string containing information from top 3 chunks
    """
    vectorstore = PineconeVectorStore(index=index, embedding=embedding_model)
    context = ""
    docs = await vectorstore.asimilarity_search(query=query, k=3)
    for doc in docs:
        context += (doc.page_content + " ")

    return context

if __name__ == "__main__":
    # context = fetch_docs(query='What is considered counterfeiting of coins, Government stamps, or currency notes, and what is the punishment?')
    # print(context)

    print("Runs successfully. No errors!")

