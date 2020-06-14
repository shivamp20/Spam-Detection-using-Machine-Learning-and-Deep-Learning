# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 00:26:48 2020

@author: Mohini Pandey
"""

from utils import get_model, int2label, label2int
from keras.preprocessing.sequence import pad_sequences

import pickle 
import numpy as np

SEQUENCE_LENGTH = 100

# get the tokenizer
    
        
tokenizer = pickle.load(open("results/tokenizer.pickle", "rb"))

model = get_model(tokenizer)
model.load_weights("results/spam_classifier_0.05")

def get_predictions(text):
    sequence = tokenizer.texts_to_sequences([text])
    # pad the sequence
    sequence = pad_sequences(sequence, maxlen=SEQUENCE_LENGTH)
    # get the prediction
    prediction = model.predict(sequence)[0]
    # one-hot encoded vector, revert using np.argmax
    return int2label[np.argmax(prediction)]

while True:
    text = input("Enter the mail:")
    # convert to sequences
    print(get_predictions(text))