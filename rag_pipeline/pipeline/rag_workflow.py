from typing import TypedDict
from langgraph.graph import StateGraph, END, START

# from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

from .data_ingestion import fetch_docs
from .prompts import query_classify_pt
from .prompts import augment_generate_prompt
from .prompts import warn_prompt
from .prompts import relevancy_prompt, irrelevant_response_prompt
from .schema_classes import QueryClassify, ResponseClassify
import os
from dotenv import load_dotenv

load_dotenv()
google_api_key = os.getenv('GOOGLE_API_KEY')

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash', api_key=google_api_key, temperature=0.8)

class AgentState(TypedDict):
    query:str
    query_type:str
    response:str

async def process_query(state:AgentState) -> AgentState:
    """Processes user query."""
    print("-> Entered node - process_query")

    struct_model = model.with_structured_output(QueryClassify)
    chain = query_classify_pt | struct_model
    query = state['query']
    res = await chain.ainvoke({'query':query})

    state_upd = {'query_type':res.query_type}
    return state_upd

def route_logic(state:AgentState) -> str:
    """Logic for conditional edge."""
    print("-> Entered route logic function - route_logic")

    query_type = state['query_type']
    if query_type == "related":
        return "law-related"
    else:
        return "not-law-related"
    
async def augment_generate(state:AgentState) -> AgentState:
    """Augments retrived docs and generates answers"""
    print("-> Entered node - augment_generate")

    chain = augment_generate_prompt | model | StrOutputParser()
    query = state['query']
    context = await fetch_docs(query=query)
    res = await chain.ainvoke({'query':query, 'context':context})
    state_upd = {'response':res}
    return state_upd

async def process_response(state:AgentState) -> AgentState:
    """Processes LLM's response. Checks if it is relevant for answering the query or not."""
    print("-> Entered node - process_response")
    print()
    
    response = state['response']
    query = state['query']
    struct_model = model.with_structured_output(ResponseClassify)
    chain = relevancy_prompt | struct_model
    chain_irr = irrelevant_response_prompt | model | StrOutputParser()

    res = await chain.ainvoke({'query':query, 'response':response})
    if res.relevancy == "relevant":
        print("-> Response is relevant")
        return {}
    else:
        print("-> Response is irrelevant")

        res_irr = await chain_irr.ainvoke({'query':query, 'response':response})
        state_upd = {'response':res_irr}
        return state_upd

async def warn_user(state:AgentState) -> AgentState:
    """Warns users to ask domain related questions."""
    print("-> Entered node - warn_user")

    chain = model | StrOutputParser()
    res = await chain.ainvoke(warn_prompt)
    state_upd = {'response':res}
    return state_upd

graph = StateGraph(AgentState)
graph.add_node(process_query)
graph.add_node(augment_generate)
graph.add_node(process_response)
graph.add_node(warn_user)
graph.add_edge(START, "process_query")
graph.add_edge("augment_generate", "process_response")
graph.add_conditional_edges(
    source="process_query",
    path=route_logic,
    path_map={'law-related':'augment_generate', 'not-law-related':'warn_user'}
)

graph.add_edge("warn_user", END)
graph.add_edge("process_response", END)
rag_app = graph.compile()