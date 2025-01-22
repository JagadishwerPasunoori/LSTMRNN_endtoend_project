import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# load LSTM model:
model = load_model('next_word_prediction.h5')

# load tokenizer:
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# function to predict next word:
def predict_next_word(seed_text,model,tokenizer,max_sequence_len):
  token_list=tokenizer.texts_to_sequences([seed_text])[0]
  if len(token_list) >= max_sequence_len:
    token_list = token_list[-(max_sequence_len-1):]
  token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
  predicted = model.predict(token_list, verbose=0)
  predicted_word_index = np.argmax(predicted, axis=1)
  for word, index in tokenizer.word_index.items():
    if index == predicted_word_index:
      return word
  return None


# Define the Streamlit app:
st.title('Next Word Prediction with LSTM RNN')
st.write('This app predicts the next word in a sentence using a LSTM RNN model.')
seed_text = st.text_input('Enter a sentence:', 'The quick brown fox')
if st.button('Predict'):
  next_word = predict_next_word(seed_text,model,tokenizer,max_sequence_len=20)
  st.write(f'The next word is: {next_word}')

