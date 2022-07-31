import json
import pickle

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer

from datasets import DATASET


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Selecting appropriate feature set in the pipeline"""

    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        if self.key == 'summary':
            arr = []
            for r in data:
                arr.append(' '.join(r.summary['stemmed']))
            return arr

        elif self.key == 'postagged':
            arr = []
            for r in data:
                arr.append(' '.join(
                    r.pos_tagged_summary['stemmed'] + r.pos_tagged_description['stemmed']))
            return arr


def multilabel_clf(train_set, test_set, src_keys):
    """Multi-label classification using MultinomialNB"""

    if len(train_set) <= 1:
        return [0] * len(src_keys)

    train_fixed = []
    for i in train_set:
        train_fixed.append(i.fixed_files)

    # Classes need to be binarized for the classifier
    mlb = MultiLabelBinarizer()
    train_labels = mlb.fit_transform(train_fixed)

    classifier = Pipeline([
        ('feats', FeatureUnion([
            ('summ', Pipeline([
                ('summary', FeatureSelector('summary')),
                ('summ_tfidf', TfidfVectorizer(sublinear_tf=True, lowercase=False))
            ])),
            ('summ_desc', Pipeline([
                ('postagged', FeatureSelector('postagged')),
                ('summ_desc_tfidf', TfidfVectorizer(
                    sublinear_tf=True, lowercase=False))
            ])),
        ])),
        ('clf', OneVsRestClassifier(MultinomialNB()))
    ])

    classifier.fit(train_set, train_labels)

    # Getting probabilities for all source files
    probas = classifier.predict_proba(test_set)

    labeled_proba = dict(zip(mlb.classes_, probas[0]))

    src_probas = []
    for name in src_keys:
        src_probas.append(labeled_proba.get(name, 0))

    return src_probas


def prepare_clf(bug_reports):
    """Preparing train set and test set based on previously fixed bugs"""

    with open(DATASET.root / 'preprocessed_src.pickle', 'rb') as file:
        src_files = pickle.load(file)

    bug_reports = list(bug_reports.values())
    # print(bug_reports)

    min_max_scaler = MinMaxScaler()

    probabilities = []
    for i, report in enumerate(bug_reports):
        probas = multilabel_clf(bug_reports[:i], [report], src_files.keys())

        some = []
        for count in probas:
            some.append(float(count))

        probas = np.array(some).reshape(-1, 1)

        normalized_probas = np.concatenate(
            min_max_scaler.fit_transform(probas)
        )

        probabilities.append(normalized_probas.tolist())

    return probabilities


def main():

    with open(DATASET.root / 'preprocessed_reports.pickle', 'rb') as file:
        bug_reports = pickle.load(file)

    probabilities = prepare_clf(bug_reports)
    # print(probabilities)

    with open(DATASET.root / 'fixed_bug_reports.json', 'w') as file:
        json.dump(probabilities, file)


if __name__ == '__main__':
    main()
