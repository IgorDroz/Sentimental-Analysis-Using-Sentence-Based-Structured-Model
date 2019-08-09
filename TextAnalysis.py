import numpy as np
import pandas as pd

from time import time
from typing import List
from collections import Counter
from sklearn.metrics import accuracy_score

"""
Feature_family is function, which given text_label and pair of sentences returns list of string, where each string 
represents active feature from this family. We need string representation to map those features to indecies.
"""

def label_features(prev_sentence, sentence, prev_label, label, text_label):
    pair_labels = ['pair_label ' + str(prev_label) + str(label) + str(text_label)]
    single_labels = ['single_label ' + str(label) + str(text_label)]
    return pair_labels + single_labels

def ngram_features(prev_sentence, sentence, prev_label, label, text_label):
    words = sentence.split(' ')
    unigram1 = ['unigram1 ' + str(prev_label) + str(label) + str(text_label) + x for x in words]
    unigram2 = ['unigram2 ' + str(label) + str(text_label) + x for x in words]
    bigram = ['bigram ' + str(prev_label) + str(label) + str(text_label) + words[i] + ' ' + words[i+1] for i in range(len(words)-1)]
    return unigram1+unigram2+bigram

all_feature_families = [label_features, ngram_features]

class Text:
    # TODO: think if we want sentence as string or list of words
    def __init__(self, sentences: List[str], sentence_probabilities: List[float], text_label):
        self.sentences = sentences
        self.label = text_label
        self.probabilities = sentence_probabilities
        self.sentence_labels = [round(x) for x in self.probabilities]
        self.feature_vector = None
        self.size = len(sentences)

class TextAnalysis:
    """
    This class learn text level model and make inference with it
    1) Go through all pairs of sentences from every text, and create feature space by looking at possible features
        1.a) Maybe count features and remove those with count below threshold
    2) Create feature representation for each text: np.array of size total_features, where i-th number is count of
    feature i in text
    3) Start learning. For each iteration:
        go over all texts. for each text:
            find argmax labeling of text using current weights vector. If it's not correct - update weights vector
    """
    def __init__(self, texts: List[Text]):
        self.texts = texts
        all_features = set()
        # Creating feature space
        for text in texts:
            prev_sentence = '*'
            prev_label = '*'
            for i in range(text.size):
                for feature_family in all_feature_families:
                    features = feature_family(prev_sentence, text.sentences[i], prev_label, text.sentence_labels[i], text.label)
                    all_features = all_features.union(features)
                prev_sentence = text.sentences[i]
                prev_label = text.sentence_labels[i]

        # assign index to each feature
        self.feature_to_index = {feature:index for index, feature in enumerate(sorted(all_features))}
        self.total_features = len(self.feature_to_index)
        self.w = np.ones(self.total_features)
        # go through all texts, and convert text.features(list of strings) to np.array
        for text in texts:
            text.feature_vector = self.get_text_feature_vector(text)

    def get_clique_feature_vector(self, sentence_1, sentence_2, label_1, label_2, text_label):
        # Given clique of sentence i-1, sentence i, text label returns vector representing clique
        features = []
        for feature_family in all_feature_families:
            features += feature_family(sentence_1, sentence_2, label_1, label_2, text_label)
        feature_vector = np.zeros(self.total_features)
        # count features, convert to np.array
        counts = Counter([self.feature_to_index[x] for x in features])
        for feature_number, count in counts.items():
            feature_vector[feature_number] = count
        return feature_vector

    def get_text_feature_vector(self, text: Text, given_labels=None):
        # given text, calculates all cliques and sums up their feature vectors
        # If given_labels provided then they are used, otherwise original text's labelings are used
        if given_labels is not None:
            text_label = given_labels[1]
            sentence_labels = given_labels[0]
        else:
            text_label = text.label
            sentence_labels = text.sentence_labels

        feature_vector = np.zeros(self.total_features)
        prev_sentence = '*'
        prev_label = '*'
        # iterate over cliques, create features
        for i in range(len(sentence_labels)):
            feature_vector += self.get_clique_feature_vector(prev_sentence, text.sentences[i], prev_label,
                                                           sentence_labels[i], text_label)
            prev_sentence = text.sentences[i]
            prev_label = text.sentence_labels[i]

        return feature_vector

    def get_text_score(self, text, given_labels=None):
        return np.dot(self.w, self.get_text_feature_vector(text, given_labels))

    def get_clique_score(self, sentence_1, sentence_2, label_1, label_2, text_label):
        return np.dot(self.w, self.get_clique_score(sentence_1, sentence_2, label_1, label_2, text_label))

    def structured_perceptron(self, num_iterations):
        # Each iteration go over texts, find argmax, update w if argmax is incorrect
        # I also added fancy penalty term
        alpha = 0.5
        for i in range(num_iterations):
            for text in self.texts:
                predicted_sentence_labels, predicted_text_label = self.viterbi(text)
                sentence_accuracy = accuracy_score(text.sentence_labels, predicted_sentence_labels)
                text_accuracy = predicted_text_label == text.label
                if sentence_accuracy != 1 or text_accuracy != 1:
                    self.w += text.feature_vector
                    penalty = alpha*(1-text_accuracy) + (1-alpha)*sentence_accuracy
                    self.w -= penalty*self.get_text_feature_vector(text, (predicted_sentence_labels, predicted_text_label))

    def viterbi(self, text: Text):
        best_score = -100000
        best_text_label = -1
        best_sentence_labels = [-1]*text.size
        for possible_text_label in [0,1]:
            scores_table = np.zeros((text.size,2))
            backtrack_table = np.zeros((text.size, 2))
            scores_table[0][0] = self.get_clique_score('*', text.sentences[0], '*', 0, possible_text_label)
            scores_table[0][1] = self.get_clique_score('*', text.sentences[0], '*', 1, possible_text_label)
            prev_sentence = text.sentences[0]
            for sentence_index in range(1,text.size):
                current_sentence = text.sentences[sentence_index]
                for possible_current_label in [0,1]:
                    scores = [scores_table[sentence_index-1][prev_label] +
                              self.get_clique_score(prev_sentence, current_sentence, prev_label, possible_current_label, possible_text_label)
                              for prev_label in [0, 1]]
                    best_prev_label = 0 if scores[0] > scores[1] else 1
                    scores_table[sentence_index][possible_current_label] = scores[best_prev_label]
                    backtrack_table[sentence_index][possible_current_label] = best_prev_label

            best_last_label = 0 if scores_table[-1][0] > scores_table[-1][1] else 1
            best_score_with_possible_text_label = scores_table[-1][best_last_label]
            if best_score_with_possible_text_label > best_score:
                best_score = best_score_with_possible_text_label
                best_text_label = possible_text_label
                best_sentence_labels[-1] = best_last_label
                for i in range(text.size-2, 0, -1):
                    best_sentence_labels[i] = backtrack_table[i][best_sentence_labels[i+1]]

        return best_sentence_labels, best_text_label


if __name__ == "__main__":
    raw_texts = ['I love dogs. I love cats', 'fuck them all. asd qwe', 'hello. kill it']
    probabilities = [[0.7, 0.7], [0.1, 0.4], [0.6, 0.1]]
    labels = [1,0,0]
    texts = [Text(raw_texts[x].split('.'), probabilities[x], labels[x]) for x in range(3)]
    TextAnalysis(texts)

