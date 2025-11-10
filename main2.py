import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load model and processors
model = load_model("rnn_news.h5")
with open("tokenizer_news.pkl1", "rb") as f:
    tokenizer = pickle.load(f)

category_names = ['World', 'Sports', 'Business', 'Sci/Tech']
max_len = 100

st.title("News Topic Prediction (RNN/LSTM)")
st.write("Paste a news article/description and predict: Sport, World, Business, Sci/Tech")

user_input = st.text_area("Paste your news article here...")

if st.button("Predict Category"):
    seq = tokenizer.texts_to_sequences([user_input])
    padded_input = pad_sequences(seq, maxlen=max_len)
    pred = np.argmax(model.predict(padded_input), axis=1)[0]
    pred_label = category_names[pred]
    st.markdown(f"**Predicted Category:** {pred_label}")
