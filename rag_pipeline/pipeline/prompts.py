from langchain_core.prompts import ChatPromptTemplate

query_classify_pt = ChatPromptTemplate.from_template(
    """
    You have a good knowledge of law.

    User query:
    {query}

    Classify the user query into 'related' and 'not-related'.
    Meaning of these strings:
    'related' - it is related to law
    'not-related' - it is not related to law
    """
)

augment_generate_prompt = ChatPromptTemplate.from_messages(
        messages=[
            # not using system messages Google Gemini model dont accept system messages
            ("human", "You are a helpful assistant who uses BNS(Bhartiya Nyaay Sanhita). Use the provided context to answer user query. BNS Context: {context}"),
            ("human", "You can reply within 150 words"),
            ("human", "{query}")
        ]
)

warn_prompt = """
    User query is not realted to your domain. Your domain is law and Bhartiya Nyaay Sanhita(BNS).
    Tell the same to the user, warn them, and politely request them to ask domain related questions.
    """
