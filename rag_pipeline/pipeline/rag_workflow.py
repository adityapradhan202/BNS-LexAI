from typing import TypedDict
from langgraph.graph import StateGraph, END, START

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

from .data_ingestion import fetch_docs
from .prompts import query_classify_pt
from .prompts import augment_generate_prompt
from .prompts import warn_prompt
from .schema_classes import QueryClassify

model = ChatOllama(model='gemma3:4b', temperature=0.8)
struct_model = model.with_structured_output(QueryClassify)

class AgentState(TypedDict):
    query:str
    query_type:str
    response:str

def process_query(state:AgentState) -> AgentState:
    """Processes user query."""
    chain = query_classify_pt | struct_model
    query = state['query']
    res = chain.invoke({'query':query})

    state_upd = {'query_type':res.query_type}
    return state_upd

def route_logic(state:AgentState) -> str:
    """Logic for conditional edge."""
    query_type = state['query_type']
    if query_type == "related":
        return "law-related"
    else:
        return "not-law-related"
    
def augment_generate(state:AgentState) -> AgentState:
    """Augments retrived docs and generates answers"""
    chain = augment_generate_prompt | model | StrOutputParser()
    query = state['query']
    context = fetch_docs(query=query)
    res = chain.invoke({'query':query, 'context':context})
    state_upd = {'response':res}
    return state_upd

def warn_user(state:AgentState) -> AgentState:
    """Warns users to ask domain related questions."""
    chain = model | StrOutputParser()
    res = chain.invoke(warn_prompt)
    state_upd = {'response':res}
    return state_upd


graph = StateGraph(AgentState)
graph.add_node(process_query)
graph.add_node(augment_generate)
graph.add_node(warn_user)
graph.add_edge(START, "process_query")
graph.add_conditional_edges(
    source="process_query",
    path=route_logic,
    path_map={'law-related':'augment_generate', 'not-law-related':'warn_user'}
)

graph.add_edge("warn_user", END)
graph.add_edge("augment_generate", END)
rag_app = graph.compile()