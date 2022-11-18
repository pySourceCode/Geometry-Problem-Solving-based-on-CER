from keras.layers import Dense, Flatten, Input
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot
import numpy as np
import os
import json
# define documents
DATA_PATH = '../texts'
with open(os.path.join(DATA_PATH), 'r') as f:
    docs = json.load(f)

# define class labels
labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
# integer encode the documents
vocab_size = 50
encoded_docs = [one_hot(d, vocab_size) for d in docs]
print(encoded_docs)

# texts 里最长的问题有46个单词
max_length = 46
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs)

# define the model
input = Input(shape=(4, ))
x = Embedding(vocab_size, 8, input_length=max_length)(input)    #这一步对应的参数量为50*8
x = Flatten()(x)
x = Dense(1, activation='sigmoid')(x)
model = Model(inputs=input, outputs=x)
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# summarize the model
print(model.summary())

# fit the model
model.fit(padded_docs, labels, epochs=100, verbose=0)
# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
loss_test, accuracy_test = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy * 100))
# test the model
test = one_hot('good',50)
padded_test = pad_sequences([test], maxlen=max_length, padding='post')
print(model.predict(padded_test))