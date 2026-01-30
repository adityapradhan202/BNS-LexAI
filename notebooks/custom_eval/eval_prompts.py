from langchain_core.prompts import ChatPromptTemplate

context_compare = ChatPromptTemplate.from_template(
    """
    You are a RAG evaluator!
    
    User query:
    {query}

    Context from search type - 1:
    {context_1}

    Context from search type - 2:
    {context_2}

    Compare the contexts with the query. Return 1 if the context from the search type - 1 is more relevant for answering the query. Or return 2 if the context from search type - 2 is more relevant for answering the query.
    """
)