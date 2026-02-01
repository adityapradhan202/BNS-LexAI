import streamlit as st
from api_req import send_request

col1, col2, col3 = st.columns(3)
with col2:
    st.image(image='static/app-logo.png', width="stretch")
with st.form("my-form"):    
    query = st.text_area(label="Enter your query here...",
             placeholder="For example:- Punishments for critcizing someone based on religion",
            )
    
    submitted = st.form_submit_button(label="Generate",type="primary")
    
if submitted:
    response = ""
    
    with st.spinner(text="Invoking RAG workflow", show_time=True):
        response = send_request(query=query)
    st.markdown(response)

with st.sidebar:
    st.header("BNS-LexAI")
    st.caption("AI-powered legal information and case understanding assistant.")
    st.markdown("Made by -> [Aditya Pradhan](https://github.com/adityapradhan202)")
    
    st.caption("Show us support by giving a star on the repository!")
    st.markdown("-> [Github Repository](https://github.com/adityapradhan202/BNS-LexAI)")