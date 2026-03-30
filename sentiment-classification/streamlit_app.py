import streamlit as st
import joblib
import os

# Load model
model_path = os.path.join("saved_models", "sentiment_classifier.pkl")
model = joblib.load(model_path)

st.title("Sentiment Classifier")

input_text = st.text_area("Enter a sentence to analyze sentiment")

if st.button("Predict"):
    if input_text.strip() != "":
        try:
            probs = model.predict_proba([input_text])[0]
            prediction = model.predict([input_text])[0]

            confidence = probs[prediction]

            label_map = {
                0: "Negative",
                1: "Neutral",
                2: "Positive"
            }

            sentiment = label_map[prediction]

            if prediction == 0:
                st.error(f"{sentiment} (Confidence: {confidence:.2f})")
            elif prediction == 1:
                st.warning(f"{sentiment} (Confidence: {confidence:.2f})")
            else:
                st.success(f"{sentiment} (Confidence: {confidence:.2f})")

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter text")