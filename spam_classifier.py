# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 00:22:35 2020

@author: Mohini Pandey
"""


import tensorflow
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from keras.callbacks import ModelCheckpoint, TensorBoard
#from keras.layers import Embedding, LSTM, Dropout, Dense
from sklearn.model_selection import train_test_split
import time
import numpy as np
import pickle
from matplotlib import pyplot

from utils import get_embedding_vectors, get_model, SEQUENCE_LENGTH, EMBEDDING_SIZE, TEST_SIZE
from utils import BATCH_SIZE, EPOCHS, int2label, label2int

def load_data():
    """
    Loads SMS Spam Collection dataset
    """
    texts, labels = [], []
    with open("F:\Final Year Project-2019-2020\Deep Learning part\Data\SpamDetectionData.txt") as f:
        for line in f:
            split = line.split()
            labels.append(split[0].strip())
            texts.append(' '.join(split[1:]).strip())
    return texts, labels



X, y = load_data()

# Text tokenization
# vectorizing text, turning each text into sequence of integers
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
# lets dump it to a file, so we can use it in testing
pickle.dump(tokenizer, open("results/tokenizer.pickle", "wb"))
# convert to sequence of integers
X = tokenizer.texts_to_sequences(X)

X = np.array(X)
y = np.array(y)

X = pad_sequences(X, maxlen=SEQUENCE_LENGTH)

y = [ label2int[label] for label in y ]
y = to_categorical(y)

print(y[0])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=7)

model = get_model(tokenizer=tokenizer)
#weight_file="weights/PreTrain_weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"

model_checkpoint = ModelCheckpoint("results/spam_classifier_{val_loss:.2f}", save_best_only=True,
                                    verbose=1)
# for better visualization
#tensorboard = TensorBoard(f"logs/spam_classifier_{time.time()}")
# print our data shapes
print("X_train.shape:", X_train.shape)
print("X_test.shape:", X_test.shape)
print("y_train.shape:", y_train.shape)
print("y_test.shape:", y_test.shape)
# train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
          batch_size=BATCH_SIZE, epochs=EPOCHS,
          callbacks=[model_checkpoint],
          verbose=1)

result = model.evaluate(X_test, y_test)

history_dict = history.history

# summarize history for accuracy
pyplot.plot(history.history['accuracy'])
pyplot.plot(history.history['val_accuracy'])
pyplot.title('model accuracy')
pyplot.ylabel('accuracy')
pyplot.xlabel('epoch')
pyplot.legend(['training', 'validation'], loc='lower right')
pyplot.show()

pyplot.clf()
acc_values = history_dict['accuracy']

epochs = range(1, len(acc_values) + 1)

val_acc_values = history_dict['val_accuracy']
pyplot.plot(epochs, acc_values, 'bo', label='Training accuracy')
pyplot.plot(epochs, val_acc_values, 'b', label='Validation accuracy')
pyplot.title('Training and validation accuracy')
pyplot.xlabel('Epochs')
pyplot.ylabel('Loss')
pyplot.legend()
pyplot.show()

# summarize history for loss
pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('model loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['training', 'validation'], loc='upper right')
pyplot.show()

# list all data in history
print(history.history.keys())


loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
pyplot.plot(epochs, loss_values, 'bo', label='Training loss')
pyplot.plot(epochs, val_loss_values, 'b', label='Validation loss')
pyplot.title('Training and validation loss')
pyplot.xlabel('Epochs')
pyplot.ylabel('Loss')
pyplot.legend()
pyplot.show()



