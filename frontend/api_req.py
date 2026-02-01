import requests
import os
from dotenv import load_dotenv

load_dotenv()
secret_key = os.getenv('BNS_LEXAI_SECRET_KEY')
base_url = os.getenv('BASE_URL')

def send_request(query:str, secret_key=secret_key):
    try:
        response = requests.post(url=base_url+"/chat-bns-lex-ai", headers={'secret-key':secret_key},
                                 json={"query":query})
        response = response.json()
    except:
        return {"status":"Some error occured while connecting to RAG api"}

    return response['rag-response']

if __name__ == "__main__":
    res = send_request(query="What is considered as counterfeiting of money?")
    print(res['rag-response'])