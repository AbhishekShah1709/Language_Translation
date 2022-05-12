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

  data, vocab, eng_tokenizer = tokenize_data(data)
  X, y = make_n_grams(data, 4)

  return data, X, y, vocab, eng_tokenizer


# In[ ]:


eng_train = load_data('/content/drive/MyDrive/europarl-corpus/train.europarl')
perp_X, X, y, eng_vocab, eng_tokenizer = preprocess_data(eng_train)


# In[ ]:


with open('/content/drive/MyDrive/tokenizer_eng.pickle', 'wb') as handle:
    pickle.dump(eng_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('/content/drive/MyDrive/tokenizer_eng.pickle', 'rb') as handle:
    eng_tokenizer = pickle.load(handle)


# In[ ]:


reverse_vocab = {}

for k,v in eng_vocab.items():
  reverse_vocab[v] = k


# In[ ]:


num_tokens = len(eng_vocab)+1
embedding_dim = 100


# In[ ]:


embedding_layer = tf.keras.layers.Embedding(num_tokens, embedding_dim, trainable=True)

def language_model():
    
    inp = tf.keras.Input(shape=(None,), dtype="int64")
    embedded_sequences = embedding_layer(inp)

    lstm = tf.keras.layers.LSTM(256)
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


model = tf.keras.models.load_model("/content/drive/MyDrive/question1_epoch10")


# In[ ]:


eng_test = load_data('/content/drive/MyDrive/europarl-corpus/test.europarl')
eng_test_data = eng_tokenizer.texts_to_sequences(eng_test)


# In[ ]:


test_X, test_y = make_n_grams(eng_test_data, 4)


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


# In[ ]:


def get_acc(preds, labels):

  tot = np.sum(preds==labels)
  return tot/len(preds)

acc = get_acc(dec_preds, test_y)
print(acc)  


# In[ ]:


def get_perplexity(sent):

  if len(sent)<=4:
    return 0, 0, 0
  else:
    cnt=0
    perp = 0

    for i in range(len(sent)-4):
      cnt+=1
      inp = sent[i:i+4]
      target = sent[i+4]
      inp = inp.reshape(1,-1)
      pred = model.predict(inp)
      val = pred[0][target]

      perp += (math.log(val)/math.log(2))
    
    return (-1*perp)/cnt, cnt, perp


# In[ ]:


tot_perp = 0
tot_cnt = 0

train_str = ""

for i, sent in enumerate(perp_X[:1000]):
  score, curr_cnt, curr_perp = get_perplexity(np.asarray(sent))
  tot_perp += curr_perp
  tot_cnt += curr_cnt
  train_str += eng_train[i].split("\n")[0]+"\t"+str(score)+"\n"

avg_perp = (-1*tot_perp)/tot_cnt
train_str += str(avg_perp)

f = open('/content/drive/MyDrive/2018101052_LM_train.txt', 'w')
f.write(train_str)
f.close()


# In[ ]:


tot_perp = 0
tot_cnt = 0

test_str = ""

for i, sent in enumerate(eng_test_data):
  score, curr_cnt, curr_perp = get_perplexity(np.asarray(sent))
  tot_perp += curr_perp
  tot_cnt += curr_cnt
  test_str += eng_test[i].split("\n")[0]+"\t"+str(score)+"\n"

avg_perp = (-1*tot_perp)/tot_cnt
test_str+=str(avg_perp)

f = open('/content/drive/MyDrive/2018101052_LM_test.txt', 'w')
f.write(test_str)
f.close()

