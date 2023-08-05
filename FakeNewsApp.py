import os
import streamlit as st
import time
from modelGPT import FakeNewsGPT

config = {
    "Headline": "/Users/iansnyder/Desktop/Projects/Transformer/config/headline-gpt2",
    "Article": "/Users/iansnyder/Desktop/Projects/Transformer/config/article-gpt2",
}
m = FakeNewsGPT(config)
st.markdown(
    "<style>body { color: green; background-color: #f0f0f0; }</style>",
    unsafe_allow_html=True,
)
st.title("Fake News App")
headline = st.text_area(
    "Enter a Headline to generate a Fake Article, Leave blank to generate a Fake Headline and Article",
    value="",
    height=50,
    help="Enter Headline content, then click the button to generate an Article!",
)
genFakeNews = st.button("Generate Fake News", type="primary")
if genFakeNews:
    with st.spinner("Generating Fake News..."):
        if headline == "" or headline == None or headline == " " or headline == "None":
            output = m.generate(return_text=True)

        else:
            output = m.generate(from_headline=headline, return_text=True)
        st.text_area("News Article", output, height=400, key="output")


with st.expander("Advanced Options"):
    st.text("Tune parameters output probabilities")
    temp = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.85)
    top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.95)
    top_k = st.slider("Top K", min_value=1, max_value=150, value=100)
    st.text("Tune parameters for length of output")
    max_new_tokens = st.slider("Max Length", min_value=100, max_value=1000, value=500)
