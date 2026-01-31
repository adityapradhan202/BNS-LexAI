from dotenv import load_dotenv
from fastapi import FastAPI
from typing import Annotated
from fastapi import Header
from pydantic import BaseModel
from pipeline import rag_app
import os

load_dotenv()
bns_lexai_key = os.getenv('BNS_LEXAI_SECRET_KEY')

app = FastAPI()

@app.post("/test-connection")
def test_api_connection(secret_key:Annotated[str, Header()]):
    if secret_key:
        if secret_key == bns_lexai_key:
            return {'status':'Connected successfully'}
        else:
            return {'status':"Couldn't connect. Wrong credentials!"}
        
    return {'status':'No key found'}

class UserData(BaseModel):
    query:str

@app.post("/chat-bns-lex-ai")
async def invoke_rag_workflow(user_data:UserData, secret_key:Annotated[str, Header()]):
    if secret_key:
        if secret_key == bns_lexai_key:
            print(f"-> Succesfully verified credentials.")
            query = user_data.query
            try:
                output = await rag_app.ainvoke({'query':query})
            except:
                return {'status':'LLM connection error!'}
            
            return {
                'rag-response':output['response']
            }
        else:
            return {'status':"Couldn't connect. Wrong credentials!"}
        
    return {'status':'No key found'}
