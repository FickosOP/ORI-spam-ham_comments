import numpy as np
# import pandas as pd


class NaiveBayesClassifier:

    def __init__(self):
        self.training_set = []
        self.labels = []

        self.testing_set = []

        self.set_positives = []
        self.set_negatives = []

        self.set_size = 0

        self.P_S1 = 0
        self.P_S2 = 0

    def fit(self, training_set, labels):
        self.labels = labels
        # split into pos and neg set
        index = 0
        for lab in labels:
            var = training_set.iloc[index]
            if lab == 1:
                for word in var.strip().split(' '):
                    self.set_positives.append(word)
                    self.training_set.append(word)
            elif lab == 0:
                for word in var.strip().split(' '):
                    self.set_negatives.append(word)
                    self.training_set.append(word)
            index += 1
        self.set_size = len(self.training_set)

    def test(self, testing_set):
        self.testing_set = testing_set
        # return list of 0 and 1
        ret_val = []
        for el in testing_set:
            p_s1, p_s2 = self._predict(el)

            p_s1 = abs(np.exp(p_s1))
            p_s2 = abs(np.exp(p_s2))
            if p_s1 > p_s2:  # always
                ret_val.append(1)
            else:
                ret_val.append(0)
        return ret_val

    def _predict(self, single_comment):
        # returns 2 float values < 1 and > 0
        prob_sum_pos = 0
        prob_sum_neg = 0
        for word in single_comment.strip().split(' '):
            p_data = self._p_in_data(word)
            if p_data == 1:
                continue
            else:
                p_pos = self._p_in_positive(word)
                p_neg = self._p_in_negative(word)
                if p_pos != 0:
                    prob_sum_pos += np.log(p_pos / p_data) + np.log(self._prob_positive())
                if p_neg != 0:
                    prob_sum_neg += np.log(self._p_in_negative(word) / p_data) + np.log(self._prob_negative())
        return prob_sum_pos, prob_sum_neg

    def _p_in_data(self, word):
        if self.training_set.count(word) == 0:
            return 1
        ret_val = self.training_set.count(word) / len(self.training_set)
        return ret_val

    def _p_in_positive(self, word):
        if self.set_positives.count(word) == 0:
            return 0
        ret_val = self.set_positives.count(word) / len(self.set_positives)
        return ret_val

    def _p_in_negative(self, word):
        if self.set_negatives.count(word) == 0:
            return 0
        ret_val = self.set_negatives.count(word) / len(self.set_negatives)
        return ret_val

    def _prob_positive(self):
        ret_val = len(self.set_positives) / self.set_size
        return ret_val

    def _prob_negative(self):
        ret_val = len(self.set_negatives) / self.set_size
        return ret_val
