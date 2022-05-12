#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import nltk
import tensorflow as tf
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import sentence_bleu


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


# eng_train = load_data('./data/ted-talks-corpus/train.en')
# fr_train = load_data('./data/ted-talks-corpus/train.fr')

eng_train = load_data('/content/drive/MyDrive/ted-talks-corpus/train.en')
fr_train = load_data('/content/drive/MyDrive/ted-talks-corpus/train.fr')


# In[ ]:


sent_len = {}

for i, line in enumerate(eng_train):
  line = line.split("\n")[0].split(" ")
  l = len(line)
  if sent_len.get(l):
    val = sent_len.get(l)
    new_val = val + [i]
    sent_len[l] = new_val
  else:
    sent_len[l] = [i]


# In[ ]:


idxs = []

for k,v in sent_len.items():
  if k>=25 and k<=30:
    idxs.extend(v)


# In[ ]:


fin_eng_train = []

for i in idxs:
  fin_eng_train.append(eng_train[i])


# In[ ]:


fin_fr_train = []

for i in idxs:
  fin_fr_train.append(fr_train[i])


# In[ ]:


eng_train = fin_eng_train
fr_train = fin_fr_train


# In[ ]:


def tokenize_data(data, max_len):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(data)
    vocab = tokenizer.word_index
    vocab_size = len(vocab) + 1
    # print(vocab_size)
    encoded_data = tokenizer.texts_to_sequences(data)
    padded_data = tf.keras.preprocessing.sequence.pad_sequences(encoded_data, maxlen=max_len, padding='post')

    return padded_data, vocab, tokenizer


# In[ ]:


eng_data, eng_vocab, eng_tokenizer = tokenize_data(eng_train, 21)
fr_data, fr_vocab, fr_tokenizer = tokenize_data(fr_train, 21)


# In[ ]:


reverse_vocab = {}


# In[ ]:


for k,v in fr_vocab.items():
    reverse_vocab[v] = k


# In[ ]:


eng_num_tokens = len(eng_vocab) + 1
fr_num_tokens = len(fr_vocab) + 1

embedding_dim = 256


# In[ ]:


embedding_layer = tf.keras.layers.Embedding(eng_num_tokens, embedding_dim, trainable=True)


# In[ ]:


def language_model():
    
    inp = tf.keras.Input(shape=(None,), dtype="float32")
    embedded_sequences = embedding_layer(inp)
    
    lstm = tf.keras.layers.LSTM(1024)
    rep = lstm(embedded_sequences)
    repeated = tf.keras.layers.RepeatVector(32)
    rep = repeated(rep)
    lstm1 = tf.keras.layers.LSTM(1024, return_sequences=True)
    rep = lstm1(rep)
    dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(fr_num_tokens, activation="softmax"))
    output = dense(rep)
    
    model = tf.keras.Model(inp, output)

    return model

def updateWeights(model, model1):
    weights_list = model.get_weights()
    weights_list1 = model1.get_weights()
    new_vec = np.zeros(weights_list[0].shape)
    new_vec = new_vec[:weights_list[0].shape[0],:]
    model.layers[0].set_weights([new_vec])

    return model

# In[ ]:


model = language_model()


# In[ ]:

model1 = tf.keras.models.load_model('../models/trained_en_lm')
model = updateWeights(model, model1)

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["acc"])
hist = model.fit(eng_data, fr_data, batch_size=64, epochs=10)


# In[ ]:


# eng_test = load_data('./data/ted-talks-corpus/test.en')
# fr_test = load_data('./data/ted-talks-corpus/test.fr')

eng_test = load_data('/content/drive/MyDrive/ted-talks-corpus/test.en')
fr_test = load_data('/content/drive/MyDrive/ted-talks-corpus/test.fr')


# In[ ]:


eng_test_data = eng_tokenizer.texts_to_sequences(eng_test)
eng_padded_data = tf.keras.preprocessing.sequence.pad_sequences(eng_test_data, maxlen=21, padding='post')

fr_test_data = fr_tokenizer.texts_to_sequences(fr_test)
fr_padded_data = tf.keras.preprocessing.sequence.pad_sequences(fr_test_data, maxlen=21, padding='post')


# In[ ]:


preds = model.predict(eng_data)


# In[ ]:


def decode_pred(preds, fr_tokenizer):
    
    fr_vocab = fr_tokenizer.word_index
    reverse_vocab = {}
    for k,v in fr_vocab.items():
        reverse_vocab[v] = k

    all_translations = []
    all_translations_idxs = []

    for i in range(preds.shape[0]):
        translation_idxs = []
        for j in range(preds.shape[1]):
            val = np.argmax(preds[i][j])
            translation_idxs.append(val)
            if val==0:
              break
        
        all_translations_idxs.append(translation_idxs)
        
        translation = []
        for val in translation_idxs:
          if val!=0:
            translation.append(reverse_vocab[val])
            
        all_translations.append(translation)

    all_translations_idxs = np.asarray(all_translations_idxs)        
    all_translations = np.asarray(all_translations)
    return all_translations, all_translations_idxs  


# In[ ]:


all_translations, wow = decode_pred(preds, fr_tokenizer)


# In[ ]:


updated_translation = []

for translation in all_translations:
  new_translation = " ".join(translation)
  updated_translation.append(new_translation)

updated_translation = np.asarray(updated_translation)


# In[ ]:


print(updated_translation[1])


# In[ ]:


def get_bleu_score(hypothesis, reference):
    
    reference = reference.split(" ")
    hypothesis = hypothesis.split(" ")
    score = sentence_bleu([reference], hypothesis)
    
    return score


# In[ ]:


scores = []
test_str = ""

for i, sent in enumerate(fr_train):
  score = get_bleu_score(updated_translation[i], sent)
  scores.append(score)
  test_str+=updated_translation[i]+"\t"+str(score)+"\n"
    # print(score)    
    # scores.append(test_str)
scores = np.asarray(scores)
val = np.mean(scores)
# print(val)
f = open('/content/drive/MyDrive/2018101052_MT1_train.txt', 'w')
f.write(test_str)
f.write(str(val))
f.close()    


# In[ ]:


scores = []
test_str = ""

for i, sent in enumerate(fr_test):
  score = get_bleu_score(updated_translation[i], sent)
  scores.append(score)
  test_str+=updated_translation[i]+"\t"+str(score)+"\n"
    # print(score)    
    # scores.append(test_str)
scores = np.asarray(scores)
val = np.mean(scores)
# print(val)
f = open('/content/drive/MyDrive/2018101052_MT1_test.txt', 'w')
f.write(test_str)
f.write(str(val))
f.close()

