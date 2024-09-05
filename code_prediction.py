import streamlit as st

st.header('Code prediction tool', divider='blue')

st.write("Select classification method:")

if st.button("Paragraphs"):
    st.switch_page("pages/Paragraphs.py")
if st.button("Sentences"):
    st.switch_page("pages/Sentences.py")