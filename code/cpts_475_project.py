"""Import"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import os
import csv
import keras
import pickle

from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM
from keras import utils
from keras.callbacks import ReduceLROnPlateau, EarlyStopping


import matplotlib.pyplot as plt
# %matplotlib inline
from collections import Counter

import nltk
from nltk.corpus import stopwords
from  nltk.stem import SnowballStemmer
import re

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

import gensim
import gensim.downloader as api
import time
import logging

# Be sure  to download stopwords from nltk
nltk.download()

"""Constants"""

# Data
DATA_FEATURES = ["target", "ids", "date", "flag", "user", "text"]
DATA_ENCODING = "ISO-8859-1"
TRAIN_PROP = 0.8

# Cleaning
TEXT_CLEANING = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

# Sentiment
POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"
DECODE = {0: "NEGATIVE", 2: "NEUTRAL", 4: "POSITIVE"}
SENTIMENT_THRESHOLDS = (0.4, 0.7)

# Model
KERAS_MODEL = "model.h5"
WORD2VEC_MODEL = "model.w2v"
TOKENIZER_MODEL = "tokenizer.pkl"
ENCODER_MODEL = "encoder.pkl"

# Gensim
W2V_SIZE = 100
W2V_WINDOW = 7
W2V_EPOCH = 32
W2V_MIN_COUNT = 10

# Keras
SEQUENCE_LENGTH = 300
EPOCHS = 8
BATCH_SIZE = 1024

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

"""Read Data"""

#print("Open file:", "C:\Users\Dawson\Desktop\WSU\CS475\FinalProject\code")
#df = pd.read_csv(r"input_cleaner.csv", encoding =DATA_ENCODING , names=DATA_FEATURES, quotechar='"', engine='python', error_bad_lines=False)
df = pd.read_csv(r"C:\Users\Dawson\Desktop\WSU\CS475\FinalProject\code\input_cleaner.csv", encoding =DATA_ENCODING , names=DATA_FEATURES, quotechar='"', engine='python', error_bad_lines=False)
def decode_sentiment(label):
    return DECODE[int(label)]
  
df.target = df.target.apply(lambda x: decode_sentiment(x))

print(len(df))
df.head(10)

target_cnt = Counter(df.target)

plt.figure(figsize=(16,8))
plt.bar(target_cnt.keys(), target_cnt.values())
plt.title("Dataset Labels Count")

"""Prepping Data"""

stop_words = stopwords.words("english")
stemmer = SnowballStemmer("english")

def cleanText(text, stem=False):
  text = re.sub(TEXT_CLEANING, ' ', str(text).lower()).strip()
  tokens = []
  for token in text.split():
    if token not in stop_words:
      if stem:
        tokens.append(stemmer.stem(token))
      else:
        tokens.append(token)
  return " ".join(tokens)

df.text = df.text.apply(lambda x:cleanText(x))

"""Split into Training and Testing Set"""

df_train, df_test = train_test_split(df, test_size=1-TRAIN_PROP, random_state=42)
print("dr_train:", len(df_train))
print("df_test:", len(df_test))

"""Build w2v Model"""

model = gensim.models.word2vec.Word2Vec(window=W2V_WINDOW, min_count=W2V_MIN_COUNT, workers=8)

docs = [_text.split() for _text in df_train.text] 
model.build_vocab(docs)

"""Train"""

# Commented out IPython magic to ensure Python compatibility.
# %%time
# model.train(docs, total_examples=len(docs), epochs=W2V_EPOCH)

model.save(r'C:\Users\Dawson\Desktop\WSU\CS475\FinalProject\code\m2v.genism')

model = gensim.models.Word2Vec.load(r"C:\Users\Dawson\Desktop\WSU\CS475\FinalProject\code\m2v.genism")

model.wv.most_similar("hate")

# Commented out IPython magic to ensure Python compatibility.
# %%time
# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(df_train.text)
# 
# vocab_size = len(tokenizer.word_index) + 1
# print("Total words", vocab_size)

# Save tokenizer
with open(r'C:\Users\Dawson\Desktop\WSU\CS475\FinalProject\code\tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Load tokenizer
with open(r'C:\Users\Dawson\Desktop\WSU\CS475\FinalProject\code\tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Commented out IPython magic to ensure Python compatibility.
# %%time
# x_train = pad_sequences(tokenizer.texts_to_sequences(df_train.text), maxlen=SEQUENCE_LENGTH)
# x_test = pad_sequences(tokenizer.texts_to_sequences(df_test.text), maxlen=SEQUENCE_LENGTH)

"""Ecode Labels"""

labels = df_train.target.unique().tolist()
labels.append(NEUTRAL)
labels

encoder = LabelEncoder()
encoder.fit(df_train.target.tolist())

y_train = encoder.transform(df_train.target.tolist())
y_test = encoder.transform(df_test.target.tolist())

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

print("y_train",y_train.shape)
print("y_test",y_test.shape)

print("x_train", x_train.shape)
print("y_train", y_train.shape)
print()
print("x_test", x_test.shape)
print("y_test", y_test.shape)

"""Embedding"""
 
embedding_matrix = np.zeros((vocab_size, W2V_SIZE))
for word, i in tokenizer.word_index.items():
  if word in model.wv:
    embedding_matrix[i] = model.wv[word]
print(embedding_matrix.shape)

embedding_layer = Embedding(vocab_size, W2V_SIZE, weights=[embedding_matrix], input_length=SEQUENCE_LENGTH, trainable=False)

"""Build Model"""

model_seq = Sequential()
model_seq.add(embedding_layer)
model_seq.add(Dropout(0.5))
model_seq.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model_seq.add(Dense(1, activation='sigmoid'))

model_seq.summary()

model_seq.compile(loss='binary_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])
callbacks = [ ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),
              EarlyStopping(monitor='val_accuracy', min_delta=1e-4, patience=5)]

# Commented out IPython magic to ensure Python compatibility.
# %%time
# history = model_seq.fit(x_train, y_train,
#                     batch_size=BATCH_SIZE,
#                     epochs=EPOCHS,
#                     validation_split=0.1,
#                     verbose=1,
#                     callbacks=callbacks)

model_seq.save(r"C:\Users\Dawson\Desktop\WSU\CS475\FinalProject\code\model.keras")

def decode_sentiment(score, include_neutral=True):
    if include_neutral:        
        label = NEUTRAL
        if score <= SENTIMENT_THRESHOLDS[0]:
            label = NEGATIVE
        elif score >= SENTIMENT_THRESHOLDS[1]:
            label = POSITIVE

        return label
    else:
        return NEGATIVE if score < 0.5 else POSITIVE

def predict(text, include_neutral=True):
    start_at = time.time()
    # Tokenize text
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=SEQUENCE_LENGTH)
    # Predict
    score = model_seq.predict([x_test])[0]
    # Decode sentiment
    label = decode_sentiment(score, include_neutral=include_neutral)

    return {"label": label, "score": float(score),
       "elapsed_time": time.time()-start_at}

predict("I love the music")

model_2 = keras.models.load_model(r'C:\Users\Dawson\Desktop\WSU\CS475\FinalProject\code\model.keras')

def predict2(text, include_neutral=True):
    start_at = time.time()
    # Tokenize text
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=SEQUENCE_LENGTH)
    # Predict
    score = model_2.predict([x_test])[0]
    # Decode sentiment
    label = decode_sentiment(score, include_neutral=include_neutral)

    return {"label": label, "score": float(score),
       "elapsed_time": time.time()-start_at}

print(predict2("I loved the music"))
print(predict2("I hate the music"))
print(predict2("I thought the music was meh"))
print(predict2("i don't know what i'm doing"))

"""Evaluate"""

# Commented out IPython magic to ensure Python compatibility.
# %%time
# score = model_seq.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
# print()
# print("ACCURACY:",score[1])
# print("LOSS:",score[0])

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
 
epochs = range(len(acc))
 
plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
 
plt.figure()
 
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
 
plt.show()

# Commented out IPython magic to ensure Python compatibility.
# %%time
# y_pred_1d = np.array([])
# y_test_1d = np.array(list(df_test.target))
# scores = model_2.predict(x_test, verbose=1, batch_size=8000)
# y_pred_1d = [decode_sentiment(score, include_neutral=False) for score in scores]
# 
# def plot_confusion_matrix(cm, classes,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
# 
#     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# 
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title, fontsize=30)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=90, fontsize=22)
#     plt.yticks(tick_marks, classes, fontsize=22)
# 
#     fmt = '.2f'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
# 
#     plt.ylabel('True label', fontsize=25)
#     plt.xlabel('Predicted label', fontsize=25)

#predict_x= (model_2.predict(x_test,batch_size=100)).astype("int32")
classes_x=np.argmax(predict_x,axis=1)
plot_confusion_matrix(model_2,predict_x, y_test)

# Commented out IPython magic to ensure Python compatibility.
# %%time
# 
# cnf_matrix = plot_confusion_matrix(y_test_1d, y_pred_1d)
# plt.figure(figsize=(12,12))
# plot_confusion_matrix(cnf_matrix, classes=df_train.target.unique(), title="Confusion matrix")
# plt.show()



print(predict2("I didn't like how it turned out."))
print(predict2("I like how it didn't turned out."))