# app.py

import streamlit as st
import pandas as pd

# Create a DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35]
}
df = pd.DataFrame(data)

def main():
    st.title('Simple Streamlit App')
    st.write('This is a simple Streamlit app with a DataFrame:')
    st.dataframe(df)

if __name__ == "__main__":
    main()
