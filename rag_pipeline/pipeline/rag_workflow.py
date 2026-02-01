from typing import TypedDict
from langgraph.graph import StateGraph, END, START

# from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

from .data_ingestion import fetch_docs
from .prompts import query_classify_pt
from .prompts import augment_generate_prompt
from .prompts import warn_prompt
from .prompts import context_relevancy_prompt
from .schema_classes import QueryClassify, ContextClassify
import os
from dotenv import load_dotenv

load_dotenv()
google_api_key = os.getenv('GOOGLE_API_KEY')

# model = ChatOllama(model='gemma3:4b', temperature=0.8)
model = ChatGoogleGenerativeAI(model='gemini-3-flash-preview', api_key=google_api_key) # with default temperature

class AgentState(TypedDict):
    query:str
    query_type:str
    context:str
    context_relevancy:str
    response:str

async def process_query(state:AgentState) -> AgentState:
    """Processes user query."""
    print("-> Entered node - process_query")

    struct_model = model.with_structured_output(QueryClassify)
    chain = query_classify_pt | struct_model
    query = state['query']
    res = await chain.ainvoke({'query':query})

    state_upd = {'query':query, 'query_type':res.query_type}
    return state_upd

def route_logic(state:AgentState) -> str:
    """Logic for conditional edge."""
    print("-> Entered route logic function - route_logic")

    query_type = state['query_type']
    if query_type == "related":
        return "law-related"
    else:
        return "not-law-related"
    
async def process_context(state:AgentState) -> AgentState:
    """Processes the context retrieved"""
    print("-> Entered node - process_context")
    query = state['query']
    context = await fetch_docs(query=query)
    struct_model = model.with_structured_output(ContextClassify)

    chain = context_relevancy_prompt | struct_model
    res = await chain.ainvoke({'query':query, 'context':context})
    state_upd = {
        'context':context,
        'context_relevancy':res.relevancy
    }

    return state_upd

def context_route_logic(state:AgentState) -> str:
    """Logic for conditional edges emerging from node - process_context"""
    print("-> Entered route logic function - context_route_logic")

    context_relevancy = state['context_relevancy']
    if context_relevancy == "relevant":
        return "answer"
    else:
        return "dont-answer"
    
async def warn_user(state:AgentState) -> AgentState:
    """Warns users to ask domain related questions."""
    print("-> Entered node - warn_user")

    chain = model | StrOutputParser()
    res = await chain.ainvoke(warn_prompt)
    state_upd = {'response':res}
    return state_upd

async def augment_generate(state:AgentState) -> AgentState:
    """Augments the context and generates an answer."""
    print("-> Entered node - augment_generate")

    query = state['query']
    context = state['context']
    chain = augment_generate_prompt | model | StrOutputParser()

    res = await chain.ainvoke({'query':query, 'context':context})
    state_upd = {'response':res}
    return state_upd

async def handle_irrelevant_context(state:AgentState) -> AgentState:
    """Handles irrelevant context"""
    print("-> Entered node - handle_irrelevant_context")

    prompt = "Tell the user that you don't have enough information to answer this query"
    chain = model | StrOutputParser()
    res = await chain.ainvoke(prompt)

    state_upd = {'response':res}
    return state_upd

graph = StateGraph(AgentState)
graph.add_node(process_query)
graph.add_node(process_context)
graph.add_node(warn_user)
graph.add_node(augment_generate)
graph.add_node(handle_irrelevant_context)

graph.add_edge(START, "process_query")
graph.add_conditional_edges(
    source="process_query",
    path=route_logic,
    path_map={'law-related':'process_context', 'not-law-related':'warn_user'}
)
graph.add_conditional_edges(
    source="process_context",
    path=context_route_logic,
    path_map={'answer':'augment_generate', 'dont-answer':'handle_irrelevant_context'}
)
graph.add_edge("warn_user", END)
graph.add_edge("augment_generate", END)
graph.add_edge("handle_irrelevant_context", END)
rag_app = graph.compile()