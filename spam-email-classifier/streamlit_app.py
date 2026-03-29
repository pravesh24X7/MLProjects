import streamlit as st
import joblib

# Load model
model = joblib.load("model/spam_email_model.pkl")

st.title("Spam Email Classifier")

input_text = st.text_area("Enter your email message")

if st.button("Predict"):
    if input_text.strip() != "":
        prediction = model.predict([input_text])[0]
        prob = model.predict_proba([input_text])[0][1]

        if prediction == 1:
            st.error(f"Spam (Confidence: {prob:.2f})")
        else:
            st.success(f"Not Spam (Confidence: {1 - prob:.2f})")
    else:
        st.warning("Please enter text")
