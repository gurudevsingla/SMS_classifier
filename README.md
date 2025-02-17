# Data

**Context**

The SMS Spam Collection is a set of SMS tagged messages that have been collected for SMS Spam research. It contains one set of SMS messages in English of 5,574 messages, tagged acording being ham (legitimate) or spam.

Dataset Link : https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

# Task

Given SMS we have to classify it into two categories whether it is Spam or Ham (Normal)

# Approach

- Used NLTK,Regex to preprocess text
- Used Count Vectorizer and TF-IDF Vectorizer to convert the documents into vectors
- Algorithms Used
  - Linear SVM
  - RBF SVM
  - Multinomial Naive Bayes
- Using Count Vectorizer/TF-IDF Vectorizer does not impact in significant difference of F1-Score so choosing simpler one i.e. Count Vectorizer
- The best F1-Score achieved using Multinomial Naive Bayes i.e. 0.92

# Deployement

Deployed on Streamlit

Link - https://sms-classifier-ml.streamlit.app/
