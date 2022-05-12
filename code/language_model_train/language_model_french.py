#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import nltk
import math
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence
import pickle


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


def load_data(path):
    
    f = open(path,'r')
    content = f.readlines()
    f.close()
    
    return content


# In[ ]:


def make_n_grams(data, n):

  all_X = []
  all_y = []

  for line in data:
    if len(line)<=n:
      continue

    else:
      for i in range(0, len(line)-n):
        X = line[i:i+n]
        y = line[i+n]

        all_X.append(np.asarray(X))
        all_y.append(y)

  all_X = np.asarray(all_X)
  all_y = np.asarray(all_y)

  return all_X, all_y


# In[ ]:


def tokenize_data(data):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(data)
    vocab = tokenizer.word_index
    vocab_size = len(vocab) + 1
    encoded_data = tokenizer.texts_to_sequences(data)

    return encoded_data, vocab, tokenizer


# In[ ]:


def preprocess_data(data):

  data, vocab, fr_tokenizer = tokenize_data(data)
  X, y = make_n_grams(data, 4)

  return X, y, vocab, fr_tokenizer


# In[ ]:


fr_train = load_data('/content/drive/MyDrive/news-crawl-corpus/train.news')
X, y, fr_vocab, fr_tokenizer = preprocess_data(fr_train)


# In[ ]:


reverse_vocab = {}

for k,v in fr_vocab.items():
  reverse_vocab[v] = k


# In[ ]:


num_tokens = len(fr_vocab)+1
embedding_dim = 100


# In[ ]:


embedding_layer = tf.keras.layers.Embedding(num_tokens, embedding_dim, trainable=True)

def language_model():
    
    inp = tf.keras.Input(shape=(None,), dtype="int64")
    embedded_sequences = embedding_layer(inp)

    lstm = tf.keras.layers.LSTM(64)
    rep = lstm(embedded_sequences)
    
    dense = tf.keras.layers.Dense(num_tokens, activation="softmax")
    output = dense(rep)

    model = tf.keras.Model(inp, output)

    return model


# In[ ]:


model = language_model()


# In[ ]:


model.summary()


# In[ ]:


model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["acc"])
model.fit(X, y, batch_size=128, epochs=10)


# In[ ]:


model.save("/content/drive/MyDrive/question1_epoch10_french")


# In[ ]:


fr_test = load_data('/content/drive/MyDrive/news-crawl-corpus/test.news')
fr_test_data = fr_tokenizer.texts_to_sequences(fr_test)


# In[ ]:


test_X, test_y = make_n_grams(fr_test_data, 4)


# In[ ]:


preds = model.predict(test_X, batch_size=16)


# In[ ]:


def decode_pred(preds):
    
    all_preds_idx = []
    all_preds = []

    for i in range(preds.shape[0]):
      val = np.argmax(preds[i])
      all_preds_idx.append(val)
      
    for val in all_preds_idx:
      all_preds.append(reverse_vocab[val])
      
    all_preds_idx = np.asarray(all_preds_idx)
    all_preds = np.asarray(all_preds)
    
    return all_preds_idx, all_preds


# In[ ]:


dec_preds, dec_preds_word = decode_pred(preds)

