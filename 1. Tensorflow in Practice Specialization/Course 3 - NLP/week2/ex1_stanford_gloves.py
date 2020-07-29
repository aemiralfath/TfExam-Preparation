import csv
import random
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

embedding_dim = 100
max_length = 16
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"
training_size = 160000
test_portion = .1

corpus = []

num_sentences = 0
with open("../../dataset/training_cleaned.csv") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        list_item = [row[5]]
        this_label = row[0]
        if this_label == '0':
            list_item.append(0)
        else:
            list_item.append(1)
        num_sentences += 1
        corpus.append(list_item)

print(num_sentences)
print(len(corpus))
print(corpus[0])

sentences = []
labels = []
random.shuffle(corpus)
for x in range(training_size):
    sentences.append(corpus[x][0])
    labels.append(corpus[x][1])

tokenizer = Tokenizer(oov_token=oov_tok)
tokenizer.fit_on_texts(sentences)

word_index = tokenizer.word_index
vocab_size = len(word_index)

sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

split = int(test_portion*training_size)

test_sequences = padded[0:split]
training_sequences = padded[split:training_size]
test_labels = labels[0:split]
training_labels = labels[split:training_size]

print(vocab_size)
print(word_index['i'])

embeddings_index = {}
with open('../../dataset/glove.6B.100d.txt') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

embeddings_matrix = np.zeros((vocab_size+1, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embeddings_matrix[i] = embedding_vector

print(len(embeddings_matrix))

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(
        vocab_size+1,
        embedding_dim,
        input_length=max_length,
        weights=[embeddings_matrix],
        trainable=False
    ),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv1D(64, 5, activation='relu'),
    tf.keras.layers.MaxPool1D(pool_size=4),
    tf.keras.layers.LSTM(64),
    # tf.keras.layers.GlobalAveragePooling1D(),
    # tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

training_padded = np.array(training_sequences)
training_labels = np.array(training_labels)
testing_padded = np.array(test_sequences)
test_labels = np.array(test_labels)

history = model.fit(training_padded, training_labels, epochs=50, validation_data=(testing_padded, test_labels))
print("Training Complete")


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()


plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')
