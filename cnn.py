import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cPickle
from collections import defaultdict
import re
from bs4 import BeautifulSoup
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import os
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder
import time
from keras import metrics
np.random.seed(10)
import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten , Dropout
from keras.layers import Conv1D,Conv2D, MaxPooling2D ,  MaxPooling1D
from keras.models import Sequential
import matplotlib.pylab as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer


def lemmatize_all(sentence):
    wnl = WordNetLemmatizer()
    for word, tag in pos_tag(word_tokenize(sentence)):
        if tag.startswith("NN"):
            yield wnl.lemmatize(word, pos='n')
        elif tag.startswith('VB'):
            yield wnl.lemmatize(word, pos='v')
        elif tag.startswith('JJ'):
            yield wnl.lemmatize(word, pos='a')
        elif tag.startswith('R'):
            yield wnl.lemmatize(word, pos='r')
            
        else:
            yield word


def msgProcessing(raw_msg):
    m_w=[]
    words2=[]
    raw_msg=str(raw_msg)
    raw_msg = str(raw_msg.lower())
    raw_msg=re.sub(r'[^a-zA-Z]', ' ', raw_msg)
    return raw_msg
    #words=raw_msg.lower().split()
    #for i in words:
    #    if len(i)>=2:
    #        words2.append(i)
    #stops=set(stopwords.words('english'))
    #m_w=" ".join([w for w in words2 if not w in stops])
    #return(" ".join(lemmatize_all(m_w)))


MAX_SEQUENCE_LENGTH = 75
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.25

batch_size = 1000
num_classes = 1
epochs = 4
ps = PorterStemmer()
print("begin golve ===")
GLOVE_DIR = "./glove.6B"
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()


# cleaning the txts


def clean_str(string):
    string = re.sub(r"\\", "", string)    
    string = re.sub(r"\'", "", string)    
    string = re.sub(r"\"", "", string)
    string = re.sub(r'[^a-zA-Z]', " ", string)
    #if string in set(stopwords.words('english')):
    #    string = " "
    return string.strip().lower()


tokenizer = Tokenizer(num_words=MAX_NB_WORDS)

def predoing(data2,y):
    texts2 = []
    labels = y[:]
    for idx in range(data2['SentimentText'][:].shape[0]):
        texts2.append(msgProcessing(nonumber(data2['SentimentText'][idx])))
    tokenizer.fit_on_texts(texts2)
    sequences = tokenizer.texts_to_sequences(texts2)
    word_index = tokenizer.word_index
    ln = len(word_index)
    print('Found %s unique tokens.' % len(word_index))
    data = sequence.pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH) 
    embedding_matrix = np.random.random((ln + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        #print(embedding_vector)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return word_index , embedding_matrix , labels , data




def nonumber(s):
    if type(s) != str:
        s = str(s) + "nothing here"
        #print(s)
        # and not word in set(stopwords.words('english'))
    
    result = ''.join([ps.stem(word) for word in s if not word.isdigit() ])
    return result


def predoing2(data2):
    texts2 = []
    for idx in range(data2['SentimentText'][:].shape[0]):
        texts2.append(msgProcessing(nonumber(data2['SentimentText'][idx])))
    sequences = tokenizer.texts_to_sequences(texts2)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    data = sequence.pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH) 
    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return data




df_train = pd.read_csv("Mtrain2.csv", error_bad_lines=False)

y = df_train.iloc[:,1].values


word_index , embedding_matrix , labels , data = predoing(df_train,y)


indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]


print y_train.sum(axis=0)
print y_val.sum(axis=0)



from keras.layers import Embedding
model= Sequential()
model.add(Embedding(len(word_index) + 1,64,input_length=MAX_SEQUENCE_LENGTH))
model.add(GlobalMaxPooling1D())
model.add(Dropout(0.1))
model.add(Dense(1864,activation='relu'))
model.add(Dropout(0.45))
model.add(Dense(120,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(23,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(num_classes, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])



# In[14]:

model.summary()

class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

history = AccuracyHistory()
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint('.mamonmdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_val, y_val),callbacks=[earlyStopping, mcp_save, reduce_lr_loss,history])

plt.plot(range(1, epochs+1), history.acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()
score = model.evaluate(x_val, y_val, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

