import streamlit as st
from modelFlanT5 import SummaryModel

# from transformers import pipeline


@st.cache_resource  # 👈 Add the caching decorator
def load_model():
    return SummaryModel()
    # return pipeline("sentiment-analysis")


# Load the model at the top
m = load_model()


def summarize_text(input_text):
    # Use the pre-loaded 'model' to generate the summary
    return m.generate(input_text)


# Title of the application
st.title("Text Summarization Interface")

# Text area for user input
user_input = st.text_area("Enter the text you want to summarize:")

# Button to trigger summarization
if st.button("Summarize"):
    if user_input:
        # Get the summarized text
        summarized_text = summarize_text(user_input)

        # Display the summarized text
        st.subheader("Summarized Text:")
        st.write(summarized_text)
    else:
        st.warning("Please enter text to summarize.")
