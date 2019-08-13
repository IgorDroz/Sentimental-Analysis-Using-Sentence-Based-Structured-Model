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
from TextAnalysis import Loader, Text
import math
class ProbabilityAnalysis:

    def __init__(self, texts: List[Text], test, method: str, granularity=4, n_gram=3, ignore_duplicates = False):
        self.texts = texts
        self.test = test
        self.method = method
        self.n_gram = n_gram
        self.granularity = granularity
        all_grams = []
        # Creating feature space
        for text in texts:
            grams = self.get_ngrams(text.probabilities, text.label)
            if ignore_duplicates:
                grams = list(set(grams))
            all_grams += grams
        self.counts = Counter(all_grams)
        self.log_probabilities = {}
        for prob_tuple, count in self.counts.items():
            other_tuple = tuple([1-prob_tuple[0]] + list(prob_tuple[1:]))
            if other_tuple in self.counts.keys():
                self.log_probabilities[prob_tuple] = math.log(count / (count + self.counts[other_tuple]))
            else:
                self.log_probabilities[prob_tuple] = math.log(count / (count + 1))
                self.log_probabilities[other_tuple] = math.log(1/(count+1))

    def get_ngrams(self, probabilities, text_label):
        result = []
        labels = [int(x*self.granularity) for x in probabilities]
        for n in range(1, self.n_gram+1):
            result += [tuple([text_label] + labels[i:i+n]) for i in range(len(labels)-n+1)]
        return result

    def get_test_accuracy(self):
        accuracy = []
        for text in self.test:
            prob_0 = sum([self.log_probabilities.get(x, 0) for x in self.get_ngrams(text.probabilities, 0)])
            prob_1 = sum([self.log_probabilities.get(x, 0) for x in self.get_ngrams(text.probabilities, 1)])
            predicted_text_label = 0 if prob_0 > prob_1 else 1
            text_accuracy = predicted_text_label == text.label
            accuracy.append(text_accuracy)
        return sum(accuracy) / len(accuracy)


if __name__ == "__main__":
    #method = 'probability_BoW_Polarity'
    results = {}
    for method in ['probability_BoW_Polarity', 'probability_GloVe_Polarity', 'probability_FastText_Polarity']:
        texts = Loader.load_texts_from_csv(method)
        print(len(texts))
        train, test = train_test_split(texts, train_size=0.8)
        for gran in range(3,15):
            for n_grams in range(3,6):
                runner = ProbabilityAnalysis(train, test, method, gran, n_grams, ignore_duplicates=True)
                acc = runner.get_test_accuracy()
                results[method, gran, n_grams] = acc
                print(method, 'gran:', gran, 'n_grams:', n_grams, 'accuracy:', acc)
    for res in sorted(results.items(), key=lambda x: x[1]):
        print(res)

