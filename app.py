import re
import string
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import streamlit as st
import pickle

nltk.download('punkt')
nltk.download('stopwords')

stop_words = stopwords.words('indonesian')
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def transform_text(spam):
    # Function to preprocess text
    spam = re.sub(r'https?://\S+|www\.\S+', '', spam)
    spam = re.sub(r'<.*?>', '', spam)
    spam = re.sub("["
                  u"\U0001F600-\U0001F64F"
                  u"\U0001F300-\U0001F5FF"
                  u"\U0001F680-\U0001F6FF"
                  u"\U0001F1E0-\U0001F1FF"
                  "]+", '', spam, flags=re.UNICODE)
    spam = re.sub(r'[0-9]+', '', spam)
    spam = re.sub(r'\$\w*', '', spam)
    spam = re.sub(r'^RT[\s]+', '', spam)
    spam = re.sub(r'#', '', spam)
    translator = str.maketrans('', '', string.punctuation)
    spam = spam.translate(translator)
    spam = spam.lower()
    tokens = word_tokenize(spam)
    with open("slangwords.txt") as f:
        kamusSlang = eval(f.read())
    pattern = re.compile(r'\b(' + '|'.join(kamusSlang.keys()) + r')\b')
    tokens = [pattern.sub(lambda x: kamusSlang[x.group()], word) for word in tokens]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

# Load model and vectorizer
tfidf = pickle.load(open('vec.pkl', 'rb'))
model = pickle.load(open('clf.pkl', 'rb'))

# Streamlit app
st.title("Klasifikasi SPAM SMS")

# Multiple file input
uploaded_files = st.file_uploader("Upload File CSV", type=["csv"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)
        if 'Pesan' in df.columns:
            st.write(f"Data yang ingin di prediksi : {uploaded_file.name}:")
            st.write(df)

            if st.button(f'Prediksi Data : {uploaded_file.name}'):
                transformed_texts = []
                predictions = []
                for index, row in df.iterrows():
                    transformed_text = transform_text(row['Pesan'])
                    vectorized_text = tfidf.transform([transformed_text])
                    vectorized_text_dense = vectorized_text.toarray()
                    prediction = model.predict(vectorized_text_dense)[0]
                    transformed_texts.append(transformed_text)
                    predictions.append(prediction)
                df['Transformed Text'] = transformed_texts
                df['Prediction'] = predictions
                st.write(f"Hasil Pemrosesan dan Prediksi dari : {uploaded_file.name}:")
                st.write(df)
        else:
            st.error(f"File CSV : {uploaded_file.name} tidak mengandung kolom 'Pesan'.")