import numpy as np
import pandas as pd
import os
from time import time, asctime
from typing import List, Dict

from collections import Counter
from sklearn.metrics import accuracy_score
import pickle
from sklearn.model_selection import train_test_split
import itertools
import logging
ALL_METHODS = ['probability_BoW_Polarity', 'probability_GloVe_Polarity', 'probability_FastText_Polarity', 'random']


"""
Feature_family is function, which given text_label and pair of sentences returns list of string, where each string 
represents active feature from this family. We need string representation to map those features to indecies.
"""
# TODO: maybe we shall think about some additional features ?
def label_features(prev_sentence, sentence, prev_label, label, text_label):
    result = []
    result.append('label1 ' + str(label) + str(text_label))
    result.append('label2 ' + str(prev_label) + str(label))
    result.append('label3 ' + str(prev_label) + str(label) + str(text_label))
    return result


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


all_feature_families = [label_features, ngram_features]


class Text:
    @staticmethod
    def load_texts_from_csv(method='random', test=False):
        backup_path = ('test' if test else 'train')+'.pkl'
        if os.path.exists(backup_path):
            with open(backup_path, 'rb') as f:
                texts = pickle.load(f)
                for text in texts:
                    text.set_method(method)
                return texts
        path = 'sentencesDSTest.csv' if test else 'sentencesDS.csv'
        df = pd.read_csv(path).set_index('text_id')
        df['random'] = np.random.uniform(0, 1, len(df))
        texts = []
        for text_id in df.index.unique():
            sub_df = df.loc[text_id]
            if len(sub_df) < 6 or type(sub_df) == pd.Series or np.any(sub_df['sentence'].isna()):
                # print('skipping', text_id)
                continue
            probabilities = {x: list(sub_df[x]) for x in ALL_METHODS}
            texts.append(Text(text_id, list(sub_df['sentence']), probabilities, sub_df['text_label'].values[0], method))
        logging.debug(' '.join(['Loaded', str(len(texts)), 'texts from csv']))
        with open(backup_path, 'wb') as f:
            pickle.dump(texts, f)
        return texts

    def __init__(self, id, sentences: List[str], sentence_probabilities: Dict[str, List[float]], text_label, method):
        self.sentences = sentences
        self.label = text_label
        self._all_probabilities = sentence_probabilities
        self.probabilities = self._all_probabilities[method]
        self.sentence_labels = [round(x) for x in self.probabilities]
        self.feature_counts = {}
        self.size = len(sentences)
        self.id = id
        self.method = method

    def set_method(self, method):
        self.method = method
        self.probabilities = self._all_probabilities[method]
        self.sentence_labels = [round(x) for x in self.probabilities]

    def get_probability(self):
        import math
        sentence_lengths = [len(sentence.split(' ')) for sentence in self.sentences]
        total_words = sum(sentence_lengths)
        sentence_scores = [math.log(p / (1 - p)) for p in self.probabilities]
        text_score = sum([sentence_scores[i] * sentence_lengths[i] for i in range(self.size)]) / total_words
        probability = math.exp(text_score) / (1 + math.exp(text_score))
        return probability

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

    def __init__(self, texts: List[Text], test, validation, method: str):
        self.texts = texts
        self.test = test
        self.validation = validation
        self.accuracies = {}
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
                logging.debug(' '.join(['finished', str(ind+1), 'texts']))
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
        # TODO: test different penalty terms
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
            logging.info(' '.join(['finished iteration', str(self.iterations), 'in', str(time()-start),
                                   'train accuracy is', str(sum(accuracy_at_i)/len(accuracy_at_i))]))
            if self.iterations%5 == 0:
                with open('vectors/' + self.method + str(self.iterations) +'.pkl', 'wb') as f:
                    pickle.dump(self.w, f)
                accuracy = self.get_accuracy()
                self.accuracies[self.iterations] = accuracy
                logging.info(' '.join([self.method, 'validation accuracy', str(accuracy)]))

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
            if best_text_label == -1 or best_score_with_possible_text_label > best_score:
                best_score = best_score_with_possible_text_label
                best_text_label = possible_text_label
                best_sentence_labels[-1] = best_last_label
                for i in range(text.size-2, -1, -1):
                    best_sentence_labels[i] = backtrack_table[i+1][best_sentence_labels[i+1]]

        return best_sentence_labels, best_text_label

    def get_accuracy(self, validation=True):
        accuracy = []
        to_test = self.validation if validation else self.test
        for text in to_test:
            predicted_sentence_labels, predicted_text_label = self.viterbi(text)
            text_accuracy = predicted_text_label == text.label
            accuracy.append(text_accuracy)
        return sum(accuracy) / len(accuracy)

    def find_best_vector(self):
        best_round = sorted(self.accuracies.items(), key=lambda x: x[1], reverse=True)[0][0]
        with open('vectors/' + self.method + str(best_round) + '.pkl', 'rb') as f:
            self.w = pickle.load(f)
        final_accuracy = self.get_accuracy(False)
        with open('results_' + self.method + '.txt', 'w') as f:
            f.write('Best round: ' + str(best_round))
            f.write('Validation accuracy: ' + str(self.accuracies[best_round]))
            f.write('Test accuracy: ' + str(final_accuracy))


def run():
    if not os.path.exists('./vectors'):
        os.mkdir('vectors')
    logging.basicConfig(level=logging.DEBUG, filename=str(asctime()).replace(':', '_').replace(' ', '_') + '.log',
                        filemode='w')
    logging.getLogger().addHandler(logging.StreamHandler())

    for method in ALL_METHODS[1:]:
        logging.info('METHOD: ' + method)
        texts = Text.load_texts_from_csv(method)
        start = time()
        # okay, this is insane shit, but without train_test_split some weird shit happens and code brokes.
        # like really, if u put texts instead of train in TextAnalysis then train accuracy is close to 1, and test is 0.
        # dafuq its 2 am, it took me 3 hours to find this out
        # TODO: try to understand what the holy fuck happened here
        train, _ = train_test_split(texts, train_size=0.9999, random_state=1)
        all_test = Text.load_texts_from_csv(method, True)
        validation, test = train_test_split(all_test, train_size=0.5, random_state=1)
        runner = TextAnalysis(train, validation, test, method)
        logging.debug(' '.join(['built feature space in', str(time() - start), 'total_features', str(runner.total_features)]))
        runner.structured_perceptron(80)
        runner.find_best_vector()


if __name__ == "__main__":
    run()