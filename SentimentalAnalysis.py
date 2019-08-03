import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from time import time
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_svmlight_files
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

class SentenceToVec(object):
	"""
	represents sentences which each word is a word2Vec to a sentence vector
	the sentence vector is made by the centroid of all words in the sentence
	"""
	def __init__(self, word2vec):
		self.word2vec = word2vec
		self.word2weight = None
		self.dim = len(next(iter(word2vec.items()))[1])

	def fit(self, X, y):
		return self

	def transform(self, X):
		return np.array([
			np.mean([self.word2vec[w] for w in words if w in self.word2vec] or
					[np.zeros(self.dim)], axis=0)for words in X])

class SentimentalAnalysis:
	def __init__(self,data_dir,level='text',representation='BoW',inference='logistic_regression'):
		self.data_dir = data_dir
		self.level = level
		self.representation = representation
		self.inference = inference

		self.load_data()
		if representation == 'BoW':
			representation_data = self.tfidf()
		elif representation == 'GloVe':
			sent2Vec = SentenceToVec(self.w2v)
			sent2Vec = sent2Vec.fit(self.data['train']['text'],self.data['train']['label'])
			representation_data = [sent2Vec.transform(self.data['train']['text']),
								   sent2Vec.transform(self.data['test']['text'])]

		training_labels, testing_labels = self.getLabels()
		training_features = representation_data[0]
		testing_features = representation_data[1]

		if inference=='logistic_regression':
			print("Logistic Regression Classifier")
			result = self.logisticReg(training_features, training_labels, testing_features, testing_labels)
			print("Accuracy = {}% , Time = {} seconds".format(result[1], result[2]))
		elif inference=='random_forest':
			print("Random Forest Classifier")
			result = self.random_forest(training_features, training_labels, testing_features, testing_labels)
			print("Accuracy = {}% , Time = {} seconds".format(result[1], result[2]))



	# def load_files(self, files):
	# 	return load_svmlight_files(files, n_features=None, dtype=None)

	def load_data(self):
		"""Loads the IMDB train/test datasets from a folder path.
		Input:
		data_dir: path to the "aclImdb" folder.

		Returns:
		train/test datasets as pandas dataframes.
		"""

		print("Loading...")
		import os
		self.data = {}
		for split in ["train", "test"]:
			self.data[split] = []
			for sentiment in ["neg", "pos"]:
				score = 1 if sentiment == "pos" else 0

				path = os.path.join(self.data_dir, split, sentiment)
				file_names = os.listdir(path)
				for f_name in file_names:
					with open(os.path.join(path, f_name), "r",encoding="utf8") as f:
						review = f.read()
						if self.level=='text':
							self.data[split].append([review, score])
						else:
							for sentence in review.split('. '):  # lines are not separated by \n
								self.data[split].append([sentence.rstrip('.'), None])

			if self.representation=='GloVe' and split=='train':
				import gensim
				from gensim.models.phrases import Phrases, Phraser
				# let X be a list of tokenized texts (i.e. list of lists of tokens)
				unigrams=list(map(lambda x: x[0].split(" "), self.data[split]))
				bigrams = Phrases(unigrams,min_count=2)
				bigram_phraser = Phraser(bigrams)
				self.word_model = gensim.models.Word2Vec(bigram_phraser[unigrams], min_count=2)
				self.w2v = dict(zip(self.word_model.wv.index2word, self.word_model.wv.syn0))

		# ToDo : tag sentences if needed
		np.random.shuffle(self.data["train"])
		self.data["train"] = pd.DataFrame(self.data["train"],columns=['text', 'label'])
		np.random.shuffle(self.data["test"])
		self.data["test"] = pd.DataFrame(self.data["test"],columns=['text', 'label'])

	def getLabels(self):
		return np.array(self.data['train']['label']),np.array(self.data['test']['label'])

	# Calculating Tf-Idf for training and testing
	def tfidf(self):
		training_data=self.data['train']
		testing_data=self.data['test']

		# Transform each text into a vector of unigram,bigram counts
		vectorizer = CountVectorizer(ngram_range=(1,2))
		training_features = vectorizer.fit_transform(training_data["text"])
		testing_features = vectorizer.transform(testing_data["text"])
		tf_transformer = TfidfTransformer()
		training_data_tfidf = tf_transformer.fit_transform(training_features)

		# .transform on the testing data which computes the TF for each review,
		# then the TF-IDF for each review using the IDF from the training data
		testing_data_tfidf = tf_transformer.transform(testing_features)
		return [training_data_tfidf ,testing_data_tfidf]


	# Train and test Logistic Regression Classifier
	@staticmethod
	def logisticReg(training_data, training_target, testing_data, testing_target):
		start = time()
		lr = LogisticRegression()
		print("Training ...")
		lr.fit(training_data, training_target)
		print("Training Done")
		print("Testing ...")
		lr_accuracy = lr.score(testing_data, testing_target) * 100
		end = time()
		return [lr, round(lr_accuracy ,2), str(round((end -start), 2))]

	# Train and test Random Forest Classifier
	@staticmethod
	def random_forest(training_data, training_target, testing_data, testing_target):
		start = time()
		print("Training ...")
		clf_forest = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, max_features='auto', max_depth=16)
		clf_forest.fit(training_data, training_target)
		print("Training Done")
		print("Testing ...")
		clf_forest_accuracy = clf_forest.score(testing_data, testing_target) * 100
		end = time()
		return [clf_forest, round(clf_forest_accuracy, 2), str(round((end - start), 2))]


if __name__ == "__main__":
	files_dir = '../data/aclImdb/'
	SentimentalAnalysis(files_dir,
						 level='text',
						 representation='BoW',
						 inference='logistic_regression')
	# SentimentalAnalysis(files_dir,
	# 					 level='text',
	# 					 representation='GloVe',
	# 					 inference='logistic_regression')

	# files = ["../data/aclImdb/train/labeledBow.feat","./data/aclImdb/test/labeledBow.feat"]
	#
	# # Load data for training_data, training_target and testing_data, testing_target
	# print("Loading Files ...")
	# #training_data, raw_training_target, testing_data, raw_testing_target = sa.load_files(files)
	# print("Done")