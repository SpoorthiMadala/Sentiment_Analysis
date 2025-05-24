import streamlit as st
from predictor import predict_sentiment

st.title("Sentiment Analysis App")
st.write("Analyze the sentiment of a review, comment, or tweet.")

text_input = st.text_area("Enter text:", height=150)

if st.button("Analyze Sentiment"):
    if text_input.strip():
        sentiment = predict_sentiment(text_input)
        st.success(f"Predicted Sentiment: **{sentiment}**")
    else:
        st.warning("Please enter some text.")
