# model.py
# Functions that define the operation of the classifier

import numpy as np
from sklearn import svm
from sklearn.cross_validation import cross_val_score
# from sklearn.grid_search import GridSearchCV
import time

# CLASSIFICATION LABELS #
NEGATIVE = -1
POSITIVE = 1


class Classifier:
    def __init__(self):
        # SVC(kernel='linear') took about 6.8 seconds to train. LinearSVC() took about 0.04 seconds.
        self.model = svm.LinearSVC()
        # self.model = svm.SVC(kernel='linear')
        # self.model = naive_bayes.BernoulliNB()

    def train(self, x, y):
        t0 = time.time()
        self.model.fit(x, y)
        t1 = time.time()
        print "Training time (sec):", t1 - t0
        print "Model parameters:"
        print self.model
        # param_grid = {'C': 1. * np.arange(1, 10)}
        # grid_search = GridSearchCV(self.model, param_grid=param_grid, cv=3, verbose=3)
        # grid_search.fit(x, y)
        print "Evaluating on training data, this should be close to 1:", self.model.score(x, y)

    def classify(self, x):
        return self.model.predict(x)

    def evaluate(self, x, y):
        # Evaluate the trained model on the given test data. Classify each
        # item from the data, and check to see whether the assigned label
        # is correct, then print overall results.
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0

        t0 = time.time()
        for (item, label) in zip(x, y):
            result = self.classify(item)
            # print result, label
            if label == POSITIVE:
                if result == label:
                    true_positive += 1
                else:
                    false_negative += 1
            else:
                if result == label:
                    true_negative += 1
                else:
                    false_positive += 1
        t1 = time.time()
        print "Evaluation time (sec):", t1 - t0

        print "TP:", true_positive
        print "FP:", false_positive
        print "TN:", true_negative
        print "FN:", false_negative

        accuracy = float(true_positive + true_negative) / \
            (true_negative + true_positive + false_negative + false_positive)

        pos_precision = float(true_positive) / (true_positive + false_positive)
        pos_recall = float(true_positive) / (true_positive + false_negative)
        pos_f = float(2 * pos_precision * pos_recall) / (pos_precision + pos_recall)

        neg_precision = float(true_negative) / (true_negative + false_negative)
        neg_recall = float(true_negative) / (true_negative + false_positive)
        neg_f = float(2 * neg_precision * neg_recall) / (neg_precision + neg_recall)

        print "=" * 30
        print "Accuracy:", accuracy
        print "Pos precision:", pos_precision
        print "Pos recall:", pos_recall
        print "Pos f-measure:", pos_f
        print "Neg precision:", neg_precision
        print "Neg recall:", neg_recall
        print "Neg f-measure:", neg_f
        # print self.model.score(x, y)

    def cv(self, x, y):
        print "=" * 30
        print "Cross validated scores"
        scores = cross_val_score(self.model, x, y, cv=10)
        print "Scores: %s" % (str(scores))
        print "Mean: %f" % np.mean(scores)
