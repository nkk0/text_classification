from collections import OrderedDict
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Embedding
from keras.models import Sequential
from nltk.probability import FreqDist
import nltk
import numpy as np
import os
import pickle
import random


SEQUENCE_LENGTH = 30

def create_embedding_layer(
	sequence_length, words_to_ints, embedding_dimension=300, glove_filename='glove/glove.840B.300d.txt'):
	print('Creating dictionary of word embeddings.')

	# Get a list of unique words
	vocab = words_to_ints.keys()

	# Get the number of unique words
	n_vocab = len(vocab)

	# Initialize our embedding matrix
	embedding_matrix = np.zeros((n_vocab+1, embedding_dimension))
	print('Embedding matrix shape:', embedding_matrix.shape)

	# Create a dictionary of all of the word vectors we'll need
	embeddings_index = {}
	with open(glove_filename) as f:
	    for line in f:
		values = line.split(' ')
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs
	print('Found {}s word vectors.'.format(len(embeddings_index)))

	# For the unique words we have, assign them their corresponding
	# word vectors  in the embeddings matrix
	number_found = 0
	for word, i in words_to_ints.items():
	    embedding_vector = embeddings_index.get(word)
	    if embedding_vector is not None:
		# Words not found in embedding index will be all 0s
		number_found += 1
		embedding_matrix[i] = embedding_vector
	print('Found {:d} glove tokens in the corpora.'.format(number_found))

	# Create the embedding layer to be used in Keras
	embedding_layer = Embedding(
	    n_vocab+1,
	    embedding_dimension,
	    weights=[embedding_matrix],
		input_length=sequence_length,
		trainable=False,
	)

	return embedding_layer

if not os.path.isfile('x.npy'):
	padded_translated_sentences = []
	random_sentences = []
	random_words = []
	clean_corpus = []

	# Create a flat list of all words
	print('Creating sentences.')
	with open('corpus.txt', 'r') as f:
	    corpus = f.read().lower()

	corpus_tokenized = nltk.tokenize.word_tokenize(corpus)
	f_dist = FreqDist(corpus_tokenized)
	least_common_words = list(filter(lambda pair: pair[1] == 1, f_dist.most_common()))
        least_common_words = set(list(zip(*least_common_words))[0])
	

        sentences = nltk.tokenize.sent_tokenize(corpus)
	sentences_filtered = []
	for sentence in sentences:
		new_sentence = []
		for word in sentence.split():
			if word in least_common_words:
				new_sentence += ['unknownword']
			elif word.isalnum():
				new_sentence += [word]
		sentences_filtered += [' '.join(new_sentence)]

        print('Tokenizing and flattening')
	sentences = [nltk.tokenize.word_tokenize(s) for s in sentences_filtered]
	flattened = [word for sequence in sentences_filtered for word in sequence]

	# Turn all words into ints
	print('Turning words into ints.')
	words_to_ints = {w: i + 1 for i, w in enumerate(OrderedDict.fromkeys(set(flattened)))}
	with open('words_to_ints.pkl','wb') as f:
		pickle.dump(words_to_ints,f)
	for sentence in sentences:
		new_sentence = []
		for token in sentence:
			new_sentence += [words_to_ints[token]]
		translated_sentences += [new_sentence]

	# Create a list of random words and shuffle it
	print('Creating list of random words.')
        random_words = [a for b in translated_sentences for a in b]
	random.shuffle(random_words)

	# Create a padded list of sentences and a padded list of random sentences
	print('Creating padded real & fake sentences.')
	padded_good_sentences = []
	padded_bad_sentences = []
	for item in translated_sentences:
		if (len(item)) <= SEQUENCE_LENGTH and item:
			number_of_zeros = SEQUENCE_LENGTH - len(item)
			padded_good_sentences += [[0] * number_of_zeros + item]

			new_sentence = []
			for _ in range(len(item)):
				new_sentence += [random_words.pop()]

			padded_bad_sentences += [[0] * (SEQUENCE_LENGTH - len(new_sentence)) + new_sentence]
	# Create X and Y pairs
	print('Creating X and Y pairs.')
	x = padded_good_sentences + padded_bad_sentences
	y = [[1, 0]] * len(padded_good_sentences) + [[0, 1]] * len(padded_bad_sentences)


	x = np.array(x)
	x = x.reshape(x.shape[0], 1, x.shape[1])
	y = np.array(y)

	try:
		np.save('x', x)
		np.save('y', y)
	except:
		import ipdb; ipdb.set_trace()

else:
	with open('words_to_ints.pkl', 'rb') as f:
		words_to_ints = pickle.load(f)
	x = np.load('x.npy')
	y = np.load('y.npy')
	x = x.reshape(x.shape[0], x.shape[2])
	model = Sequential()
	model.add(create_embedding_layer(SEQUENCE_LENGTH, words_to_ints))
	model.add(LSTM(128, return_sequences=False, recurrent_dropout=0.2))
	model.add(Dense(2, activation='softmax'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	print(model.summary())
	filepath = 'models/model-50seq-{epoch:02d}-{val_acc:.3f}'
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [checkpoint]
	model.fit(x, y, epochs=130, callbacks=callbacks_list, validation_split=0.2)
	import ipdb; ipdb.set_trace()
