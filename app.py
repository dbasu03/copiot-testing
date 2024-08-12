# app.py

import streamlit as st
import pandas as pd
from transformers import pipeline
import torch

# Set up a pre-trained transformer model for text classification
@st.cache(allow_output_mutation=True)
def load_model():
    classifier = pipeline('sentiment-analysis')
    return classifier

# Load model
model = load_model()

# Streamlit app
def main():
    st.title('Text Classification with Transformers')
    st.write('Enter some text to get a sentiment classification:')
    
    # Text input
    user_input = st.text_area("Input Text", "Type your text here...")
    
    if st.button('Classify'):
        if user_input:
            # Predict sentiment
            predictions = model(user_input)
            st.write(f"Predictions: {predictions}")
        else:
            st.write("Please enter some text.")

if __name__ == "__main__":
    main()
