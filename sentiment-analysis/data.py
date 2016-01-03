# data.py
# Functions for processing the data set

import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import cross_validation

# CLASSIFICATION LABELS #
NEGATIVE = -1
POSITIVE = 1


def split_data(data, labels, size):
    train_x, test_x, train_y, test_y = cross_validation.train_test_split(data, labels, test_size=size, random_state=100)
    return train_x, test_x, train_y, test_y


def combine_data(pos_data, neg_data):
    # Combine the positive and negative data into a single list
    data = pos_data + neg_data
    # Create a list containing the label for each piece of data
    labels = (len(pos_data) * [POSITIVE]) + (len(neg_data) * [NEGATIVE])

    # Combine data and labels into one list of tuples so we don't lose
    # the correspondence between them when we shuffle them.
    zipped = zip(data, labels)
    # Shuffle the list; a seed is used for reproducibility.
    random.seed(290470)
    random.shuffle(zipped)
    # Separate the lists again
    data, labels = zip(*zipped)

    return data, labels


def vectorize_strings(train_data, test_data, train_labels):
    # The strings need to be transformed to a numeric vector representation
    # to allow us to actually use them for any learning tasks.
    vectorizer = TfidfVectorizer(encoding='latin-1', min_df=2, max_df=0.5, sublinear_tf=True, use_idf=True,
                                 ngram_range=(1, 2), norm='l2')
    train_vecs = vectorizer.fit_transform(train_data)
    test_vecs = vectorizer.transform(test_data)

    # Select the k most informative features and discard all of the others
    info = SelectKBest(score_func=chi2, k=10000)
    train_vecs = info.fit_transform(train_vecs, train_labels)
    test_vecs = info.transform(test_vecs)

    # print vectorizer.stop_words_
    return train_vecs, test_vecs


def import_data():
    # Open the data files; this assumes that the corpora are stored in the
    # directory 'rt-polaritydata' inside of the directory this file is in.
    # pos_data and neg_data are lists of sentences. The data is pre-tokenized.
    pos_file = './rt-polaritydata/rt-polarity.pos'
    neg_file = './rt-polaritydata/rt-polarity.neg'
    with open(pos_file, 'r') as f:
        pos_data = f.read().splitlines()
    with open(neg_file, 'r') as f:
        neg_data = f.read().splitlines()

    (data, labels) = combine_data(pos_data, neg_data)

    # Split the data into train/test sets
    (train_data, test_data, train_labels, test_labels) = split_data(data, labels, 0.2)

    # Vectorize data
    (train_vecs, test_vecs) = vectorize_strings(train_data, test_data, train_labels)

    # print train_data
    # print type(test_data)

    return train_vecs, train_labels, test_vecs, test_labels

# Debug
if __name__ == "__main__":
    import_data()

