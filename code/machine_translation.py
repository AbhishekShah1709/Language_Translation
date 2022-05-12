import numpy as np
import pickle
import tensorflow as tf

def decode_pred(preds, fr_tokenizer):

    fr_vocab = fr_tokenizer.word_index
    reverse_vocab = {}
    for k,v in fr_vocab.items():
        reverse_vocab[v] = k

    translation_idxs = []
    preds = preds[0]
    for i in range(preds.shape[0]):
        val = np.argmax(preds[i])
        translation_idxs.append(val)
        if val==0:
            break
        
    translation = translation_idxs
    translation = []
    for val in translation_idxs:
        if val!=0:
            translation.append(reverse_vocab[val])
        
    translation = np.asarray(translation)
    return translation

def translate_sent(data, fr_tokenizer):
    model = tf.keras.models.load_model('../models/model.h5')
    preds = model.predict(data.reshape(1,-1))
    translation = decode_pred(preds, fr_tokenizer)
    translation = " ".join(translation)
    print("")
    print("-------------------------------------------")
    print("French Translation:", end=" ")
    print(translation)
    print("-------------------------------------------")

def preprocess_sent(sent):

    with open('../models/tokenizer_eng2.pickle', 'rb') as handle:
        eng_tokenizer = pickle.load(handle)
    
    with open('../models/tokenizer_fr2.pickle', 'rb') as handle:
        fr_tokenizer = pickle.load(handle)

    sent_data = eng_tokenizer.texts_to_sequences(sent)
    padded_data = tf.keras.preprocessing.sequence.pad_sequences(sent_data, maxlen=21, padding='post')
    translate_sent(np.asarray(padded_data[0]), fr_tokenizer)

sent = input("Input Sentence: ")
preprocess_sent([sent])
