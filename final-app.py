# app.py
"""
Streamlit app for Sentiment Analysis using a pre-trained Logistic Regression model.
Users can upload their own model files and input text to get sentiment predictions.
"""

import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


st.set_page_config(page_title="Sentiment Analysis App", layout="centered")
st.title("💬 Sentiment Analysis with Logistic Regression")
st.write("Upload your model files and enter a review to predict whether it's **Positive** or **Negative** sentiment.")

# ------------------- Upload Model Files -------------------
st.sidebar.header("📂 Upload Models")


# Example: vector_model.pkl, model.pkl
vectorizer_file = st.sidebar.file_uploader("vector_model.pkl", type=["pkl"])
model_file = st.sidebar.file_uploader("model.pkl", type=["pkl"])


if vectorizer_file and model_file:
    model_vectorizer = pickle.load(vectorizer_file)
    sentiment_model = pickle.load(model_file)
    st.sidebar.success("✅ Models loaded successfully!")
else:
    st.sidebar.warning("Please upload both the vectorizer and model files to proceed.")
    st.stop()

# ------------------- Text Input Area -------------------
st.header("🧠 Enter Review Text for Prediction")

user_input = st.text_area("Enter a review here (e.g., 'The movie was amazing!')")

# ------------------- Preprocessing Function -------------------
def preprocess_text(document: str):
    """Clean and preprocess the input review text."""
    ps = PorterStemmer()
    review = re.sub('[^a-zA-Z0-9]', ' ', document)
    review = review.lower()
    review = review.split()
    all_stopwords = stopwords.words('english')
    if 'not' in all_stopwords:
        all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if word not in set(all_stopwords)]
    cleaned_review = ' '.join(review)
    return cleaned_review

# ------------------- Predict Button -------------------
if st.button("🔍 Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review before prediction.")
    else:
        with st.spinner("Analyzing sentiment..."):
            try:
                # Preprocess
                cleaned_text = preprocess_text(user_input)
                # Vectorize
                X = model_vectorizer.transform([cleaned_text])
                # Predict
                y_pred = sentiment_model.predict(X)

                # Interpret result
                sentiment = "😊 Positive Review" if y_pred == 1 else "😠 Negative Review"

                st.success(f"**Prediction:** {sentiment}")
                st.balloons()

            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ------------------- Footer -------------------
st.markdown("---")
st.caption("Developed with ❤️ using Streamlit | Logistic Regression | NLP Sentiment Analysis")
