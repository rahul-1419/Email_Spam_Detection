import streamlit as st
import pickle
import string
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


import spacy
from nltk.stem import PorterStemmer

nlp = spacy.load("en_core_web_sm")
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    doc = nlp(text)

    lemmatized = []

    for token in doc:
        if token.text.isalnum() and not token.is_stop and not token.is_punct:
            lemmatized.append(token.lemma_)

    # Apply stemming
    stemmed = [ps.stem(word) for word in lemmatized]

    return " ".join(stemmed)

tfidf = pickle.load(open('./model/vectorizer.pkl','rb'))
model = pickle.load(open('./model/model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")