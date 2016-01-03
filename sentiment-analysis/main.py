# main.py
# The main program; primary interface to the classification model

import data
import model

(train_data, train_labels, test_data, test_labels) = data.import_data()
classifier = model.Classifier()
classifier.train(train_data, train_labels)
classifier.evaluate(test_data, test_labels)
classifier.cv(test_data, test_labels)
