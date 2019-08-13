import numpy as np
import pandas as pd
import os
from time import time
from typing import List
from collections import Counter
from sklearn.metrics import accuracy_score
import pickle
from sklearn.model_selection import train_test_split
import itertools
"""
Feature_family is function, which given text_label and pair of sentences returns list of string, where each string 
represents active feature from this family. We need string representation to map those features to indecies.
"""


def label_features(prev_sentence, sentence, prev_label, label, text_label):
    pair_labels = ['pair_label ' + str(prev_label) + str(label) + str(text_label)]
    single_labels = ['single_label ' + str(label) + str(text_label)]
    return pair_labels + single_labels


def ngram_features(prev_sentence, sentence, prev_label, label, text_label):
    words = [x.lower() for x in sentence.split(' ') if len(x)]
    result = []
    for x in words:
        result.append('unigram1 ' + str(text_label) + x)
        result.append('unigram2 ' + str(label) + str(text_label) + x)
        result.append('unigram3 ' + str(label) + x)
        result.append('unigram4 ' + str(prev_label) + str(label)+x)
    for i in range(len(words) - 1):
        result.append('bigram1 ' + str(text_label) + words[i] + ' ' + words[i+1])
        result.append('bigram2 ' + str(label) + str(text_label) + words[i] + ' ' + words[i+1])
        result.append('bigram3 ' + str(label) + words[i] + ' ' + words[i+1])
        result.append('bigram4 ' + str(prev_label) + str(label) + words[i] + ' ' + words[i+1])
    return result


all_feature_families = [ngram_features]


class Text:
    # TODO: think if we want sentence as string or list of words
    def __init__(self, id, sentences: List[str], sentence_probabilities: List[float], text_label):
        self.sentences = sentences
        self.label = text_label
        self.probabilities = sentence_probabilities
        self.sentence_labels = [round(x) for x in self.probabilities]
        self.feature_counts = {}
        self.size = len(sentences)
        self.id = id


class TextAnalysis:

    @staticmethod
    def get_analyzer(method):
        backup_path = 'runner_' + method + '.pkl'
        if os.path.exists(backup_path):
            with open(backup_path, 'rb') as f:
                runner = pickle.load(f)
                if sum(runner.w) == 0:
                    runner.w, runner.iterations = Loader.load_vector(runner)
                return runner
        texts = Loader.load_texts_from_csv(method)
        print(len(texts))
        start = time()
        train, test = train_test_split(texts, train_size=0.8)
        runner = TextAnalysis(train, test, method)
        print('built feature space in', time()-start, runner.total_features)
        with open(backup_path, 'wb') as f:
           pickle.dump(runner, f)
        return runner
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
    def __init__(self, texts: List[Text], test, method: str):
        self.texts = texts
        self.test = test
        self.method = method
        texts_f = []
        # Creating feature space
        for ind, text in enumerate(texts):
            f = []
            prev_sentence = '*'
            prev_label = '*'
            for i in range(text.size):
                for feature_family in all_feature_families:
                    features = feature_family(prev_sentence, text.sentences[i], prev_label, text.sentence_labels[i], text.label)
                    f += features
                prev_sentence = text.sentences[i]
                prev_label = text.sentence_labels[i]
            texts_f.append(f)
            if ind%1000 == 0:
                print('finished', ind+1, 'texts')
        all_features = list(itertools.chain.from_iterable(texts_f))
        usage_counts = Counter(all_features)
        # assign index to each feature
        all_features = {x for x in all_features if usage_counts[x] > 10}
        self.feature_to_index = {feature:index for index, feature in enumerate(sorted(all_features))}
        self.total_features = len(self.feature_to_index)
        self.w = np.zeros(self.total_features)
        self.iterations = 0
        # go through all texts, and convert text.features(list of strings) to dict feature_index: count
        for i, text in enumerate(texts):
            text.feature_counts = self.features_to_count(texts_f[i])

    def features_to_count(self, features):
        feature_counts = Counter([self.feature_to_index.get(x, None) for x in features])
        if None in feature_counts.keys():
            feature_counts.pop(None)
        return feature_counts

    def get_clique_features(self, sentence_1, sentence_2, label_1, label_2, text_label):
        # Given clique of sentence i-1, sentence i, text label returns vector representing clique
        features = []
        for feature_family in all_feature_families:
            features += feature_family(sentence_1, sentence_2, label_1, label_2, text_label)
        return features

    def get_text_feature_counts(self, text: Text, given_labels=None):
        # given text, calculates all cliques and sums up their feature vectors
        # If given_labels provided then they are used, otherwise original text's labelings are used
        if given_labels is not None:
            text_label = given_labels[1]
            sentence_labels = given_labels[0]
        else:
            text_label = text.label
            sentence_labels = text.sentence_labels

        features = []
        prev_sentence = '*'
        prev_label = '*'
        # iterate over cliques, create features
        for i in range(len(sentence_labels)):
            features += self.get_clique_features(prev_sentence, text.sentences[i], prev_label,
                                                       sentence_labels[i], text_label)
            prev_sentence = text.sentences[i]
            prev_label = sentence_labels[i]
        return self.features_to_count(features)

    def get_clique_score(self, sentence_1, sentence_2, label_1, label_2, text_label):
        counts = self.features_to_count(self.get_clique_features(sentence_1, sentence_2, label_1, label_2, text_label))

        return sum([count*self.w[index] for index, count in counts.items()])

    def update_from_counts(self, counts, sign):
        for index, count in counts.items():
            if sign == '+':
                self.w[index] += count
            else:
                self.w[index] -= count

    def structured_perceptron(self, num_iterations, starting_iteration=0):
        # Each iteration go over texts, find argmax, update w if argmax is incorrect
        # I also added fancy penalty term
        alpha = 0.5
        for i in range(num_iterations):
            start = time()
            accuracy_at_i = []
            for text in self.texts:
                predicted_sentence_labels, predicted_text_label = self.viterbi(text)
                sentence_accuracy = accuracy_score(text.sentence_labels, predicted_sentence_labels)
                text_accuracy = predicted_text_label == text.label
                accuracy_at_i.append(text_accuracy)
                if sentence_accuracy != 1 or text_accuracy != 1:
                    self.update_from_counts(text.feature_counts, '+')
                    penalty = self.get_text_feature_counts(text, (predicted_sentence_labels, predicted_text_label)) #alpha*(1-text_accuracy) + (1-alpha)*sentence_accuracy
                    self.update_from_counts(penalty, '-')
            self.iterations += 1
            print('finished iteration', self.iterations, 'in', time()-start, 'train accuracy is', sum(accuracy_at_i)/len(accuracy_at_i))
            if self.iterations%5 == 0:
                with open('vectors/' + self.method + str(self.iterations) +'.pkl', 'wb') as f:
                    pickle.dump(self.w, f)
                print('test accuracy', self.get_test_accuracy())

    def viterbi(self, text: Text, test=False):
        best_score = -100000
        best_text_label = -1
        best_sentence_labels = [-1]*text.size
        for possible_text_label in [0,1]:
            scores_table = np.zeros((text.size,2))
            backtrack_table = np.zeros((text.size, 2), dtype=int)
            scores_table[0][0] = self.get_clique_score('*', text.sentences[0], '*', 0, possible_text_label)
            scores_table[0][1] = self.get_clique_score('*', text.sentences[0], '*', 1, possible_text_label)
            prev_sentence = text.sentences[0]
            for sentence_index in range(1,text.size):
                current_sentence = text.sentences[sentence_index]
                for possible_current_label in [0,1]:
                    # TODO: add penalty for test_set, based on differnce between current_label and probability of sentence
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
                for i in range(text.size-2, -1, -1):
                    best_sentence_labels[i] = backtrack_table[i+1][best_sentence_labels[i+1]]

        return best_sentence_labels, best_text_label

    def get_test_accuracy(self):
        accuracy = []
        for text in self.test:
            predicted_sentence_labels, predicted_text_label = self.viterbi(text)
            text_accuracy = predicted_text_label == text.label
            accuracy.append(text_accuracy)
        return sum(accuracy) / len(accuracy)

class Loader:
    @staticmethod
    def load_texts_from_csv(method):
        backup_path = 'texts_'+method+'.pkl'
        if os.path.exists(backup_path):
            with open(backup_path, 'rb') as f:
                return pickle.load(f)
        path = 'sentencesDS.csv'
        df = pd.read_csv(path).set_index('text_id')
        texts = []
        for text_id in df.index.unique():
            x = df.loc[text_id]
            if len(x) < 6 or type(x) == pd.Series or np.any(x['sentence'].isna()):
                # print('skipping', text_id)
                continue
            texts.append(Text(text_id, list(x['sentence']), list(x[method]), x['text_label'].values[0]))
        with open(backup_path, 'wb') as f:
            pickle.dump(texts, f)
        return texts

    @staticmethod
    def load_vector(runner):
        return np.zeros(runner.total_features), 0



if __name__ == "__main__":
    method = 'probability_BoW_Polarity'
    runner = TextAnalysis.get_analyzer(method)
    runner.structured_perceptron(100)

