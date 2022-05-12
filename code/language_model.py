import numpy as np
import math
import tensorflow as tf
import pickle
import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

def load_data(path):
    
    f = open(path,'r')
    content = f.readlines()
    f.close()
    
    return content

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

def get_perplexity(sent):

  if len(sent)<=4:
    return 0, 0, 0
  else:
    cnt=0
    perp = 0
    model = tf.keras.models.load_model('../models/trained_en_lm')

    for i in range(len(sent)-4):
      cnt+=1
      inp = sent[i:i+4]
      target = sent[i+4]
      inp = inp.reshape(1,-1)
      pred = model.predict(inp)
      val = pred[0][target]

      perp += (math.log(val)/math.log(2))
    
    return (-1*perp)/cnt, cnt, perp

def preprocess_sent(sent):

    with open('../models/tokenizer_eng.pickle', 'rb') as handle:
        eng_tokenizer = pickle.load(handle)

    sent_data = eng_tokenizer.texts_to_sequences(sent)
    score, cnt, perp = get_perplexity(np.asarray(sent_data[0]))
#    val = -1*(score*cnt)
    prob = math.pow(2, perp)
    print("")
    print("-----------------------------------------")
    print("Log perplexity: " + str(score))
    print("Probability: " + str(prob))
    print("-----------------------------------------")

sent = input("Input Sentence: ")
preprocess_sent([sent])
