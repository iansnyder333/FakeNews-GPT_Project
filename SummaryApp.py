import streamlit as st

# from transformers import pipeline


# Assume this function interacts with your model and returns the summarized text
# @st.cache_resource  # 👈 Add the caching decorator
def load_model():
    return "model"
    # return pipeline("sentiment-analysis")


# Load the model at the top
model = load_model()


def summarize_text(input_text):
    # Use the pre-loaded 'model' to generate the summary
    # summarized_text = model.generate_summary(input_text)
    # return summarized_text
    return input_text


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
