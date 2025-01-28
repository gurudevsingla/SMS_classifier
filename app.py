import streamlit as st
import joblib
from utils import clean_tokenized_sentence
import nltk

# Download necessary NLTK data
nltk.download('punkt_tab')
nltk.download('stopwords')

# Page Title
st.title("📩 Spam Message Classifier")
st.markdown(
    """
    Enter a message below to determine if it's **Spam** or **Not Spam**. 
    This classifier uses a trained machine learning model to analyze your input.
    """
)

# Input Area
st.write("### 📝 Enter Your Message:")
user_input = st.text_area(
    "Type your message here...",
    height=150,
    placeholder="E.g., 'You have won a $1000 Walmart gift card! Claim now!'",
)

# Load the model and CountVectorizer
try:
    clf_loaded = joblib.load('./models/mnb_model.pkl')
    f_loaded = joblib.load('./models/count_vectorizer.pkl')
except FileNotFoundError:
    st.error("⚠️ Model files not found. Ensure the trained model and vectorizer are in the `models/` directory.")
    st.stop()

# Predict Button
if st.button("🔍 Predict"):
    if user_input.strip():  # Ensure input is not empty
        msg_cleaned = clean_tokenized_sentence(user_input)
        if msg_cleaned:
            # Process and classify the input
            X_new = f_loaded.transform([msg_cleaned])
            prediction = clf_loaded.predict(X_new)
            prediction_result = "Spam" if prediction[0] == 1 else "Not Spam"

            # Display Result
            st.write(f"### Prediction: **{prediction_result}**")
            if prediction_result == "Spam":
                st.warning("⚠️ This message is likely **Spam**.")
            else:
                st.success("✅ This message is likely **Not Spam**.")
        else:
            st.info("🚫 Could not process the message. Try entering a more descriptive text.")
    else:
        st.error("❌ Please enter a message before clicking 'Predict'.")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | Model: Multinomial Naive Bayes")
