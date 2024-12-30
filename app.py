import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load the pre-trained model and vectorizer

with open("spam_classifier_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Streamlit App
st.title("Spam Email/SMS Classifier")
st.write("Enter an email/sms below to check if it's Spam or Not.")

# Input Text Box
user_input = st.text_area("Email Content", "Type your email content here...")

if st.button("Classify"):
    if user_input.strip() == "":
        st.error("Please enter some content!")
    else:
        # Preprocess and transform user input
        input_transformed = vectorizer.transform([user_input]).toarray()
        prediction = model.predict(input_transformed)
        result = "Spam" if prediction[0] == 1 else "Not Spam"
        
        # Display the result
        st.write(f"The email is classified as: **{result}**")
