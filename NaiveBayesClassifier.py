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

        self.P1 = 0
        self.P2 = 0

    def fit(self, training_set, labels):
        """
        Creating sets of positive and negative comments. Depending on class one comment is tokenized and all tokens
        are appended to corresponding set
        :param training_set: List of all comments, prepared for training
        :param labels: List of labels for each comment(1-SPAM, 0-HAM)
        """
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

        prob_positive = labels.value_counts()
        self.P2 = prob_positive[0] / (prob_positive[0] + prob_positive[1])
        self.P1 = prob_positive[1] / (prob_positive[0] + prob_positive[1])

    def test(self, testing_set):
        """
        Predicts class for every comment is set
        :param testing_set: List of comments
        :return: List of predicted classes
        """
        self.testing_set = testing_set
        # return list of 0 and 1
        ret_val = []
        for el in testing_set:
            p_s1, p_s2 = self._predict(el)

            if p_s1 > 0 > p_s2:
                ret_val.append(1)
                continue

            elif p_s2 > 0 > p_s1:
                ret_val.append(0)
                continue

            p_s1 = abs(np.exp(p_s1))
            p_s2 = abs(np.exp(p_s2))
            if p_s1 > p_s2:
                ret_val.append(1)
            else:
                ret_val.append(0)
        return ret_val

    def _predict(self, single_comment):
        """
        Predicts class for single comment
        :param single_comment: one comment from set
        :return: probability that it is spam and probability that it is ham
        """
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
        """
        :param word: One word from comment
        :return: Ratio of how many times word appeared in training set and how many words are in training set
        """
        if self.training_set.count(word) == 0:
            return 1
        ret_val = self.training_set.count(word) / len(self.training_set)
        return ret_val

    def _p_in_positive(self, word):
        """
        :param word: One word from comment
        :return: Ratio of how many times word appeared in set of positive comments and how many words are in that set
        """
        if self.set_positives.count(word) == 0:
            return 0
        ret_val = self.set_positives.count(word) / len(self.set_positives)
        return ret_val

    def _p_in_negative(self, word):
        """
        :param word: One word from comment
        :return: Ratio of how many times word appeared in set of negative comments and how many words are in that set
        """
        if self.set_negatives.count(word) == 0:
            return 0
        ret_val = self.set_negatives.count(word) / len(self.set_negatives)
        return ret_val

    def _prob_positive(self):
        return self.P1

    def _prob_negative(self):
        return self.P2
