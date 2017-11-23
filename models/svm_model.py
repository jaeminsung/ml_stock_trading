# -*- encoding: utf-8 -*-
import os
import sys
curDir = os.path.dirname(__file__)
sys.path.append('{0}/../scripts/'.format(curDir))
from model import Model
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

class SVMModel(Model):

    def __init__(self):
        self.attributes = None
        self.selected_attr = None
        self.model = None
        self.train_data = None

    def feature_transform(self, data):
        # Feature transformation: scale data using stanardization
        scaler = StandardScaler().fit(data)
        return scaler.transform(data)

    def feature_select(self, X_data, Y_data):
        # L1-based feature selection
        lsvc = svm.LinearSVC(C=0.01, penalty="l1", dual=False).fit(X_data, Y_data)
        m = SelectFromModel(lsvc, prefit=True)

        important_attr = m.get_support(indices=True)
        print("")
        print("Selected attributes: ")
        for i in range(len(important_attr)):
            print(self.attributes[important_attr[i]])
        print("")
        self.selected_attr = important_attr

        return m.transform(X_data)

    def preprocess(self, X_data, Y_data):
        X_transformed = self.feature_transform(X_data)
        X_selected = self.feature_select(X_transformed, Y_data)
        return X_selected

    def train(self, X_df, Y):
        self.train_data = X_df

        # Apply preprocessing to X
        self.attributes = X_df.columns
        X = self.preprocess(X_df, Y)

        X = np.array(X)

        # Train/Test Split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=42)

        self.model = svm.SVC(kernel='rbf')
        self.model.fit(X_train, Y_train)
        score = self.model.score(X_test, Y_test)

        return score

    def predict(self, X_df):
        merged_df = self.train_data.append(X_df)
        X = self.feature_transform(merged_df)
        X = np.array(X[len(self.train_data):])
        # X = np.array(self.feature_transform(X_df))

        X = X[:, self.selected_attr]
        return self.model.predict(X)
