import streamlit as st
from langchain.prompts import PromptTemplate

st.write("hello")
st.write([1,2,3,4])
st.write({"x":1})
st.selectbox(
    "Choose your model",
    (
        "GPT-3",
        "GPT-4",
    ),
)

st.write(PromptTemplate)