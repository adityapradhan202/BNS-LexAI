from pipeline import rag_app

response = rag_app.invoke({'query':'How much money has the government alloted for the ration food?'})
print(response)
print()
response = rag_app.invoke({'query':'Is it punishable to criticize someone based on the religion?'})
print(response)