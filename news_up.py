# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import pickle
import pandas as pd
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# +
#mod=pickle.load(open('model2.pkl','rb'))
# -

import spacy
nlp=spacy.load('en_core_web_sm')

from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

import re
from nltk.stem import SnowballStemmer
stem = SnowballStemmer("english")
lem=WordNetLemmatizer()

## Tfidf Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# +
st.title("True or Fake Text Classification")

st.write("Enter a text to determine if it is True or Fake.")

# Get user input
user_input = st.text_area("Input Text")
def clean_words(text):
    # Define a regex pattern for URLs
    url_pattern = re.compile(r'http[s]?://\S+')
    
    # Remove URLs from the text
    text = re.sub(url_pattern, '', text)
    doc= nlp(text)
    clean_text=[x.lemma_ for x in doc if not x.is_stop and not x.is_punct and not x.like_num and not x.is_bracket and not x.pos_ in ['SYM']]
    clean_text=[stem.stem(x) for x in clean_text]
    return ' '.join(clean_text)

try :
    mod=pickle.load(open('model.pkl','rb'))
    vect=pickle.load(open('vector.pkl','rb'))
except Exception as e :
    st.error(str(e))

if len(user_input) != 0:
    df = pd.DataFrame({'text':user_input}, index=[0]) 
    df['text'] = df['text'].apply(clean_words)
    new_data = vect.transform(df['text'])   

if st.button('Predict') :
    prediction = mod.predict(new_data) 
    if prediction[0] == 1 :
        st.error('News is Real ')
    else :
        st.success('News is Fake ')

# -


