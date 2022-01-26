# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 07:54:13 2021

@author: mjvat
"""

import sys
from os.path import exists
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split


class Classifier:

    __slots__ = ('X', 'y', 'data', 'train_X', 'train_y', 'test_X', 'test_y')

    def __init__(self, csv_filepath=None, target_column=None):
        self.X = None
        self.y = None
        self.data = None
        self.train_X, self.test_X = None, None
        self.train_y, self.test_y = None, None
        self._load_data(csv_file=csv_filepath, target_column=target_column, col_headers=['A', 'B', 'label'])

    def _load_data(self, csv_file=None, target_column=None, col_headers=None):

        if not isinstance(csv_file, str):
            raise TypeError(f"{csv_file} should be {str}")

        if target_column is not None and not isinstance(target_column, str):
            raise TypeError(f"{target_column} should be {str}")

        if not exists(csv_file):
            raise IOError(f"{csv_file} file path does not exist")

        self.data = pd.read_csv(csv_file, header=col_headers if not col_headers else 0)

        # Setting last column as target column
        if target_column is None:
            target_column = self.data.columns[len(self.data.columns) - 1]

        self.y = self.data[target_column]
        self.data.drop(columns=[target_column], inplace=True)
        self.X = self.data

        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(self.X, self.y,
                                                                                test_size=0.4,
                                                                                random_state=42,
                                                                                stratify=self.y)

    def _get_best_model_and_accuracy(self, model_prefix=None, model=None, scoring='f1_score', **kwargs):
        grid = GridSearchCV(model, param_grid=kwargs.get('kwargs'), cv=5, scoring=scoring, n_jobs=-1)
        grid.fit(X=self.train_X, y=self.train_y)

        # Evaluating Predictions on Test Data
        test_score = grid.score(self.test_X, self.test_y)

        return model_prefix, grid.best_score_, test_score

    def _svm_linear(self, model_prefix='svm_linear', scoring='f1_score', **kwargs):

        svm_linear = SVC()
        return self._get_best_model_and_accuracy(model_prefix=model_prefix,
                                                 model=svm_linear,
                                                 scoring=scoring,
                                                 kwargs=kwargs)

    def _svm_polynomial(self, model_prefix='svm_polynomial', scoring='f1_score', **kwargs):
        svm_poly = SVC()
        return self._get_best_model_and_accuracy(model_prefix=model_prefix,
                                                 model=svm_poly,
                                                 scoring=scoring,
                                                 kwargs=kwargs)

    def _svm_rbf(self, model_prefix='svm_rbf', scoring='f1_score', **kwargs):
        svm_rbf = SVC()
        return self._get_best_model_and_accuracy(model_prefix=model_prefix,
                                                 model=svm_rbf,
                                                 scoring=scoring,
                                                 kwargs=kwargs)

    def _logistic(self, model_prefix='logistic', scoring='f1_score', **kwargs):
        logistic_regression = LogisticRegression()
        return self._get_best_model_and_accuracy(model_prefix=model_prefix,
                                                 model=logistic_regression,
                                                 scoring=scoring,
                                                 kwargs=kwargs)

    def _knn(self, model_prefix='knn', scoring='f1_score', **kwargs):
        knn = KNeighborsClassifier()
        return self._get_best_model_and_accuracy(model_prefix=model_prefix,
                                                 model=knn,
                                                 scoring=scoring,
                                                 kwargs=kwargs)

    def _decision_tree(self, model_prefix='decision_tree', scoring='f1_score', **kwargs):
        decision_tree = DecisionTreeClassifier()
        return self._get_best_model_and_accuracy(model_prefix=model_prefix,
                                                 model=decision_tree,
                                                 scoring=scoring,
                                                 kwargs=kwargs)

    def _random_forest(self, model_prefix='random_forest', scoring='f1_score', **kwargs):
        random_forest = RandomForestClassifier()
        return self._get_best_model_and_accuracy(model_prefix=model_prefix,
                                                 model=random_forest,
                                                 scoring=scoring,
                                                 kwargs=kwargs)

    def train_all_classifiers(self, output_csv_file='output.csv'):
        classifier_scores = []

        def _to_csv_row(x):
            return ','.join(map(str, x))

        classifier_scores.append(_to_csv_row(self._svm_linear(scoring='accuracy',
                                                              C=[0.1, 0.5, 1, 5, 10, 50, 100], kernel=('linear',))))
        classifier_scores.append(_to_csv_row(self._svm_polynomial(scoring='accuracy',
                                                                  C=[0.1, 1, 3], degree=[4, 5, 6], gamma=[0.1, 0.5],
                                                                  kernel=('poly',))))
        classifier_scores.append(_to_csv_row(self._svm_rbf(scoring='accuracy',
                                                           C=[0.1, 0.5, 1, 5, 10, 50, 100],
                                                           gamma=[0.1, 0.5, 1, 3, 6, 10],
                                                           kernel=('rbf',))))
        classifier_scores.append(_to_csv_row(self._logistic(scoring='accuracy', C=[0.1, 0.5, 1, 5, 10, 50, 100])))
        classifier_scores.append(_to_csv_row(self._knn(scoring='accuracy', n_neighbors=[val for val in range(1, 51)],
                                                       leaf_size=[val for val in range(5, 65, 5)])))
        classifier_scores.append(_to_csv_row(self._decision_tree(scoring='accuracy',
                                                                 max_depth=[val for val in range(1, 51)],
                                                                 min_samples_split=[val for val in range(2, 11)])))
        classifier_scores.append(_to_csv_row(self._random_forest(scoring='accuracy',
                                                                 max_depth=[val for val in range(1, 51)],
                                                                 min_samples_split=[val for val in range(2, 11)])))
        # print(classifier_scores)

        if output_csv_file is not None:
            try:
                with open(output_csv_file, 'w') as output_csv:
                    output_csv.writelines(row + '\n' for row in classifier_scores)
            except Exception as e:
                print(e)


def main():

    #if len(sys.argv) < 3:
    #    print("python3 problem2.py input2.csv output2.csv")
    #    sys.exit(1)

    #input_csv = str(sys.argv[1])
    #output_csv = str(sys.argv[2])
    output_csv = 'output37.csv'
    input_csv = 'E:/Transfer/Desktop/Schools/Columbia/Python Projects/Project 3 - Machine Learning/input3.csv'
    classifier = Classifier(csv_filepath=input_csv, target_column='label')
    classifier.train_all_classifiers(output_csv_file=output_csv)


if __name__ == '__main__':
    main()