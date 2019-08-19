import numpy as np
import pandas as pd
import os
from time import time
from typing import List
from collections import Counter
from sklearn.metrics import accuracy_score
import pickle
from sklearn.model_selection import train_test_split, KFold
import itertools
from TextAnalysis import Text, ALL_METHODS
import math
import logging
from time import asctime

class ProbabilityAnalysis:
    def __init__(self, texts: List[Text], method: str, granularity=4, n_gram=3, ignore_duplicates=False,
                 strict_ngrams=False):
        self.method = method
        self.n_gram = n_gram
        self.granularity = granularity
        self.strict_ngrams = strict_ngrams
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
        if self.strict_ngrams:
            n = self.n_gram
            result += [tuple([text_label] + labels[i:i+n]) for i in range(len(labels)-n+1)]
        else:
            for n in range(1, self.n_gram+1):
                result += [tuple([text_label] + labels[i:i+n]) for i in range(len(labels)-n+1)]
        return result

    def get_test_accuracy(self, test):
        accuracy = []
        for text in test:
            prob_0 = sum([self.log_probabilities.get(x, 0) for x in self.get_ngrams(text.probabilities, 0)])
            prob_1 = sum([self.log_probabilities.get(x, 0) for x in self.get_ngrams(text.probabilities, 1)])
            predicted_text_label = 0 if prob_0 > prob_1 else 1
            text_accuracy = predicted_text_label == text.label
            accuracy.append(text_accuracy)
        return sum(accuracy) / len(accuracy)

def run_cv(texts):
    texts = np.array(texts)
    results = []
    for method in ALL_METHODS:
        for text in texts:
            text.set_method(method)

        kf = KFold(n_splits=5, shuffle=True, random_state=1)
        i = 0
        for train_index, test_index in kf.split(texts):
            train = texts[train_index]
            test = texts[test_index]
            for ignore_duplicates in [True, False]:
                # If ignore duplicates is True then only single n-gram of each type from single text is accounted
                for strict_ngrams in [True, False]:
                    # If strict ngrams is True then only n-grams are accounted, else all k-grams are accounted for k<=n
                    for granularity in range(12, 15):
                        for n_grams in range(1, 6):
                            runner = ProbabilityAnalysis(train, method, granularity, n_grams,
                                                         ignore_duplicates=ignore_duplicates,
                                                         strict_ngrams=strict_ngrams)
                            acc = runner.get_test_accuracy(test)
                            results.append([method, granularity, n_grams, ignore_duplicates, strict_ngrams, i, acc])
                            logging.info([method, granularity, n_grams, ignore_duplicates, strict_ngrams, i, acc])
            logging.info(method + ' finished iter ' + str(i))
            i += 1
    df = pd.DataFrame(results, columns=['method', 'granularity', 'n_grams', 'ignore_duplicates', 'strict', 'cv', 'acc'])
    df.to_csv('prob_cv_results.csv', index=False)

def load_creative_test(method):
    backup_path = 'creative.pkl'
    if os.path.exists(backup_path):
        with open(backup_path, 'rb') as f:
            texts = pickle.load(f)
            for text in texts:
                text.set_method(method)
            return texts
    texts = []
    for path in ['sentencesCreativeDSTest.csv', 'sentencesCreativeDSTrain.csv']:
        df = pd.read_csv(path).set_index('text_id')
        df['random'] = np.random.uniform(0, 1, len(df))
        for text_id in df.index.unique():
            sub_df = df.loc[text_id]
            if len(sub_df) < 6 or type(sub_df) == pd.Series or np.any(sub_df['sentence'].isna()):
                # print('skipping', text_id)
                continue
            probabilities = {x: list(sub_df[x]) for x in ALL_METHODS}
            texts.append(Text(text_id, list(sub_df['sentence']), probabilities, sub_df['text_label'].values[0], method))
    with open(backup_path, 'wb') as f:
        pickle.dump(texts, f)
    return texts

def run():
    method = 'probability_GloVe_Polarity'
    train_texts = Text.load_texts_from_csv(method)
    train_texts += Text.load_texts_from_csv(method, test=True)
    #run_cv(train_texts)

    test_texts = load_creative_test('probability_BoW_Polarity')

    for granularity in range(2, 10):
        analyzer = ProbabilityAnalysis(train_texts, method, granularity, 1, True, False)
        logging.info('granularity ' + str(granularity) + ' gave accuracy: ' + str(analyzer.get_test_accuracy(test_texts))
                     + ' self accuracy: ' + str(analyzer.get_test_accuracy(train_texts)))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, filename=str(asctime()).replace(':', '_').replace(' ', '_') + '.log',
                        filemode='w')
    logging.getLogger().addHandler(logging.StreamHandler())
    #run()
    bbc = load_creative_test('probability_BoW_Polarity')



