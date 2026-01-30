from pipeline import rag_app

response = rag_app.invoke({'query':'What is considered as a rape?'})
print(response)