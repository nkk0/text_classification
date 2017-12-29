import random
from collections import OrderedDict
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


sentences = [['a', 'good', 'way', 'to', 'lose', 'dont', 'weight'], ['dont', 'eat', 'to', 'so', 'much', 'fat', 'fuck']]
SEQUENCE_LENGTH = 10

padded_sentences = []
padded_translated_sentences = []
random_sentences = []
random_words = []

# Create a flat list of all words
flattened = [word for sequence in sentences for word in sequence]

# Replace rare words with '<unk>'
for sentence in sentences:
	for i, word in enumerate(sentence):
		if flattened.count(word) == 1:
			sentence[i] = '<unk>'

# Turn all words into ints
flattened = [word for sequence in sentences for word in sequence]
words_to_ints = {w: i + 1 for i, w in enumerate(OrderedDict.fromkeys(set(flattened)))}

for sentence in sentences:
	new_sentence = []
	for token in sentence:
		new_sentence += [words_to_ints[token]]
	padded_translated_sentences += [new_sentence]

# Create a list of random words and shuffle it
for items in padded_translated_sentences:
	for item in items:
		if item != '.' and item != '!' and item != '?':
			random_words += [item] * random.randint(1, 2)
random.shuffle(random_words)

# Create a padded list of sentences and a padded list of random sentences
for item in padded_translated_sentences:
	if len(item) <= SEQUENCE_LENGTH:
		number_of_zeros = SEQUENCE_LENGTH - len(item)
		padded_sentences += [[0] * number_of_zeros + item]

	new_sentence = []
	for _ in range(len(item)):
		new_sentence += [random_words.pop()]

	random_sentences += [[0] * (SEQUENCE_LENGTH - len(new_sentence)) + new_sentence]

# Create X and Y pairs
x = padded_sentences + random_sentences
y = [[1]] * len(padded_sentences) + [[0]] * len(random_sentences)

import numpy as np
x = np.array(x)
x = x.reshape(x.shape[0], 1, x.shape[1])
y = np.array(y)

model = Sequential()
model.add(LSTM(8, input_shape=(None, x.shape[2])))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(x, y)
import ipdb; ipdb.set_trace()
