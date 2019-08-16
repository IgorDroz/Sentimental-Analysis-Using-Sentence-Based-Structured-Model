import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_svmlight_files
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
import re
import os
from nltk import sent_tokenize
from gensim.models import FastText
from nltk.util import ngrams
from nltk.stem import PorterStemmer
import gensim
from gensim.models.phrases import Phrases, Phraser


class SentenceToVec(object):
	"""
	represents sentences which each word is a word2Vec to a sentence vector
	the sentence vector is made by the centroid of all words in the sentence
	"""
	def __init__(self,modelName,model):
		self.model = model
		self.word2weight = None
		self.modelName=modelName

	def transform(self, X):
		if self.modelName=='GloVe':
			self.dim = len(next(iter(self.model.items()))[1])
		else:
			self.dim=300 # default for fasttext

		results=[]
		for sentences in X:
			result = np.mean([self.model[w] for w in sentences.split(" ") if w in self.model], axis=0)
			# if (isinstance(result,np.ndarray) and bool(np.isnan(result).any())) or \
			# 		(isinstance(result,float) and np.isnan(result)):
			if (isinstance(result,float) and np.isnan(result)):
				result=np.zeros(self.dim)
			results.append(result)
		print("passed")
		return np.array(results)


def tagSentences(data,domain,representation):
	if domain=='imdb':
		files_dir = '../data/aclImdb/'
	else:
		files_dir = '../data/rt-polaritydata/'

	sentenceSA = SentimentalAnalysis(files_dir,
									 level='text',
									 representation=representation,
									 inference='logistic_regression',
									 shuffle=False)

	sentenceSA.data['test']=data['train']
	predictedLabels = sentenceSA.classify(Print=False)

	for idx in range(len(data['train'].values)):
		data['train'].iloc[idx,1]=predictedLabels[idx]

	return sentenceSA.probabilities


class SentimentalAnalysis:
	def __init__(self,data_dir,level='text',representation='BoW',inference='logistic_regression',shuffle=True,infer=False):
		self.data_dir = data_dir
		self.level = level
		self.representation = representation
		self.inference = inference
		self.shuffle=shuffle
		self.infer=infer
		self.load_data()
		if infer:
			self.classify()


	def load_files(self, files):
		return load_svmlight_files(files, n_features=None, dtype=None)

	def load_model(self):
		print("Loading model...")
		if self.representation=='fasttext':
			model_path = "../crawl-300d-2M-subword.bin"
			self.word_model = FastText.load_fasttext_format(model_path,encoding='utf-8')

		if self.representation=='GloVe':
			from gensim.models import KeyedVectors
			model_path = "../glove.27B.100d.word2vec.txt"
			self.word_model = KeyedVectors.load_word2vec_format(model_path)
		self.w2v = dict(zip(self.word_model.wv.index2word, self.word_model.wv.syn0))

	def load_data(self):
		"""Loads the IMDB train/test datasets from a folder path.
		Input:
		data_dir: path to the "aclImdb" folder.

		Returns:
		train/test datasets as pandas dataframes.
		"""

		print("Loading data...")
		self.data = {}
		for split in ["train", "test"]:
			self.data[split] = []
			for sentiment in ["neg", "pos"]:
				score = 1 if sentiment == "pos" else 0

				path = os.path.join(self.data_dir, split, sentiment)
				# if split=='train':
				# 	path = os.path.join('../data/rt-polaritydata/', split, sentiment)
				file_names = os.listdir(path)
				# counter = 0
				for f_name in file_names:
					with open(os.path.join(path, f_name), "r" ,encoding="utf8") as f:
						## for debug purposes: ##

						# counter+=1
						# if counter==3:
						# 	break

						#########################

						review = f.read().lower()
						if self.level=='text':
							if '<' in review:
								review = re.sub(r'<[^>]+>', '', review)
							review = re.sub(r'[^\w\s]', ' ', review)
							review = review.replace(' s ', ' ')
							review = review.replace(' t ', 't ')
							review = re.sub(' +', ' ', review).rstrip(' ')
							self.data[split].append([review, score ,f_name , None])

						else:
							for idx,sentence in enumerate(sent_tokenize(review)):  # lines are not separated by \n
								if '<' in sentence:
									sentence = re.sub(r'<[^>]+>','',sentence)
								sentence = re.sub(r'[^\w\s]', ' ', sentence)
								sentence = sentence.replace(' s ',' ')
								sentence = sentence.replace(' t ', 't ')
								sentence = re.sub(' +',' ',sentence).rstrip(' ')
								if len(sentence)<2:
									continue
								self.data[split].append([sentence, None , f_name , str(idx).zfill(5)])

		if self.shuffle==True:
			np.random.shuffle(self.data["train"])
			np.random.shuffle(self.data["test"])
		self.data["train"] = pd.DataFrame(self.data["train"],columns=['text', 'label' , 'file_name' , 'sentence_id'])
		self.data["test"] = pd.DataFrame(self.data["test"],columns=['text', 'label' , 'file_name' , 'sentence_id'])

		if self.level=='sentence':
			self.probabilities = tagSentences(self.data,domain='other',representation=self.representation)


	def getLabels(self):
		return np.array(self.data['train']['label']),np.array(self.data['test']['label'])

	def preprocess(self):
		from nltk import word_tokenize
		print("Starting to preprocess...")
		for split in ['train','test']:
			unigrams = [word_tokenize(sentence[0]) for sentence in self.data[split].values]
			ps = PorterStemmer()
			for idx,review in enumerate(unigrams):
				stemmedSentence=[]
				for word in review:
					#stemmedSentence.append(ps.stem(word)) # stemming takes too long ...
					stemmedSentence.append(word)
				self.data[split].iloc[idx,0]=" ".join(stemmedSentence)

		bigrams = Phrases(unigrams, min_count=2)
		bigram_phraser = Phraser(bigrams)
		if self.representation == 'GloVe':
			# let X be a list of tokenized texts (i.e. list of lists of tokens)
			self.word_model = gensim.models.Word2Vec(bigram_phraser[unigrams], min_count=1)
			self.w2v = dict(zip(self.word_model.wv.index2word, self.word_model.wv.syn0))
		elif self.representation == 'fasttext':
			self.word_model = FastText(bigram_phraser[unigrams], min_count=1)
			self.w2v=dict(zip(self.word_model.wv.index2word, self.word_model.wv.syn0))


		print("Finished preprocessing.")

	def classify(self,Print=True):
		#self.preprocess()
		if self.representation == 'BoW':
			representation_data = self.tfidf() # tfidf is actually makes BoW representatiton atm
		else:
			self.load_model()
			sent2Vec = SentenceToVec(self.representation,self.w2v)
			representation_data = [sent2Vec.transform(self.data['train']['text']),
								   sent2Vec.transform(self.data['test']['text'])]

		self.training_labels, self.testing_labels = self.getLabels()
		self.training_features = representation_data[0]
		self.testing_features = representation_data[1]

		if Print:
			if self.inference=='logistic_regression':
				print("Logistic Regression Classifier")
				result = self.logisticReg(self.training_features, self.training_labels, self.testing_features, self.testing_labels)
				print("Accuracy = {}%".format(result[1]))
			elif self.inference=='random_forest':
				print("Random Forest Classifier")
				result = self.random_forest(self.training_features, self.training_labels, self.testing_features, self.testing_labels)
				print("Accuracy = {}%".format(result[1]))
		else:
			if self.inference=='logistic_regression':
				result = self.logisticReg(self.training_features, self.training_labels, self.testing_features, self.testing_labels)
				self.probabilities = result[1][:,1]
			elif self.inference=='random_forest':
				result = self.random_forest(self.training_features, self.training_labels, self.testing_features, self.testing_labels)
		print("Testing done")
		return result[0]

	# Calculating Tf-Idf for training and testing
	def tfidf(self):
		training_data=self.data['train']
		testing_data=self.data['test']

		# Transform each text into a vector of unigram,bigram counts
		vectorizer = CountVectorizer(binary=True)
		training_features = vectorizer.fit_transform(training_data["text"])
		testing_features = vectorizer.transform(testing_data["text"])
		# vectorizer = CountVectorizer()
		# tf_transformer = TfidfTransformer()
		# training_data_tfidf = tf_transformer.fit_transform(training_features)
		#
		# # .transform on the testing data which computes the TF for each review,
		# # then the TF-IDF for each review using the IDF from the training data
		# testing_data_tfidf = tf_transformer.transform(testing_features)
		# return [training_data_tfidf ,testing_data_tfidf]
		#return [training_features.toarray(), testing_features.toarray()]
		return [training_features, testing_features]


	# Train and test Logistic Regression Classifier
	@staticmethod
	def logisticReg(training_data, training_target, testing_data, testing_target):
		lr = LogisticRegression()
		print("Training ...")
		lr.fit(training_data, training_target)
		print("Training Done")
		print("Testing ...")
		predictedLabels = lr.predict(testing_data)
		probabilities = lr.predict_proba(testing_data)
		if testing_target[0] is not None:
			lr_accuracy = lr.score(testing_data, testing_target) * 100
			return [predictedLabels, round(lr_accuracy ,2),lr]
		return [predictedLabels, probabilities, lr]

	# Train and test Random Forest Classifier
	@staticmethod
	def random_forest(training_data, training_target, testing_data, testing_target):
		print("Training ...")
		clf_forest = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, max_features='auto', max_depth=16)
		clf_forest.fit(training_data, training_target)
		print("Training Done")
		print("Testing ...")
		predictedLabels = clf_forest.predict(testing_data)
		if testing_target[0] is not None:
			lr_accuracy = clf_forest.score(testing_data, testing_target) * 100
			return [predictedLabels, round(lr_accuracy ,2), clf_forest]
		return [predictedLabels, None, None]


if __name__ == "__main__":
	files_dir = '../data/aclImdb/'
	sa = SentimentalAnalysis(files_dir,
							level='text',
							representation='BoW',
							inference='random_forest',
							infer=True)

	# print("starting saBow")
	#
	# saBow = SentimentalAnalysis(files_dir, #<----- this just creates a sentence level model with tagged sentences without inference
	# 					 level='sentence',
	# 					 representation='BoW',
	# 					 inference='logistic_regression',
	# 					shuffle=False)
	#
	# print("starting saGlove")
	#
	# saGlove = SentimentalAnalysis(files_dir, #<----- this just creates a sentence level model with tagged sentences without inference
	# 					 level='sentence',
	# 					 representation='GloVe',
	# 					 inference='logistic_regression',
	# 					shuffle=False)
	#
	# print("starting saFastText")
	#
	# saFasttext = SentimentalAnalysis(files_dir, #<----- this just creates a sentence level model with tagged sentences without inference
	# 					 level='sentence',
	# 					 representation='fasttext',
	# 					 inference='logistic_regression',
	# 					shuffle=False)
	#
	# #### making the csv #####
	# header = ['text_id','sentence_id' , 'probability_BoW_Polarity' ,'probability_GloVe_Polarity' ,
	# 		  'probability_FastText_Polarity' , 'text_label' ,'sentence']
	#
	# finalResults =[]
	# for idx,sentence in enumerate(saBow.data['train'].values):
	# 	row = [sentence[-2] , sentence[-1] , saBow.probabilities[idx] , saGlove.probabilities[idx] ,
	# 		   saFasttext.probabilities[idx] , sa.data['train'].loc[sa.data['train']['file_name'] == sentence[-2]].iloc[0,1]
	# 		, sentence[0]]
	# 	finalResults.append(row)
	# data = pd.DataFrame(finalResults, columns=header)
	# data.to_csv('sentencesDS.csv', index=False, mode='w')

	SentimentalAnalysis(files_dir,
							level='text',
							representation='GloVe',
							inference='random_forest',
							infer=True)

	# SentimentalAnalysis(files_dir,
	# 						level='text',
	# 						representation='fasttext',
	# 						inference='logistic_regression',
	# 						infer=True)

	# files = ["../data/aclImdb/train/labeledBow.feat","../data/aclImdb/test/labeledBow.feat"]
	#
	# # Load data for training_data, training_target and testing_data, testing_target
	# print("Loading Files ...")
	# training_data, raw_training_target, testing_data, raw_testing_target = sa.load_files(files)
	# print("Done")