import numpy as np
import operator as op
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel

#Contains most of preprocessing tecniques
from MLClass import MLClass

#Functions for Test1
from randomModelTest1 import RandomModelTest1

#Functions for Test2
from pipeModelTest2 import ModelTest2

#Functions for Test3
from gridSearchTest3 import ModelTest3

#Functions for Test4
from votingClassifierTest4 import ModelTest4

#LOAD DATASET

#list of dataset attributes
names = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','class']

importFileTraining = pd.read_csv('./adult.data', names = names)
importFileTesting = pd.read_csv('./adult.test', names=names)

#PREPROCESSING

mlc = MLClass()
importFrame = mlc.pipePreprocessing(importFileTraining, importFileTesting, names, unknownStrategy = 'most_frequent')

data = importFrame.loc[:,names[:-1]].values
target = importFrame.loc[:,names[-1]].values

xTrain, xTest, yTrain, yTest = train_test_split(data, target, train_size = 0.7, test_size = 0.3)

classifiers = ["Logistic Regression", "Decision Tree Classifier", "LinearSVC"]

pipeLR = Pipeline([('stds', StandardScaler()),
                   ('pca', PCA(n_components = 2)),
                   ('lr', LogisticRegression())])

pipeDTree = Pipeline([('feature_selection', SelectFromModel(LinearSVC(penalty = 'l2', dual = False))),
                      ('dTree', DecisionTreeClassifier())])

pipeLSVC = Pipeline([('stds', StandardScaler()),
                     ('pca', PCA(n_components = 2)),
                     ('lsvc', LinearSVC())])

pipe = [pipeLR, pipeDTree, pipeLSVC]

#EVALUATION

#Test1
test1 = RandomModelTest1()
test1.mainRandomModel(xTrain, yTrain, xTest, yTest, classifiers, unknownStrategy = "most_frequent")

#Test2
test2 = ModelTest2()
test2.mainPipeModel(pipe, data, target, classifiers, unknownStrategy = "most_frequent")

#Test3
test3 = ModelTest3()
test3 = test3.mainGridSearchCV(pipe, data, target, xTrain, yTrain, classifiers, unknownStrategy = 'mostFrequent')

lrP1 = ModelTest3.lrParams[1]["lr__C"]
lrP2 = ModelTest3.lrParams[1]["lr__solver"]
dTreeP1 = ModelTest3.dTreeParams[1]["dTree__criterion"]
dTreeP2 = ModelTest3.dTreeParams[1]["dTree__splitter"]
dTreeP3 = ModelTest3.dTreeParams[1]["dTree__max_depth"]
lsvcP1 = ModelTest3.lsvcParams[1]["lsvc__C"]

#Test4

test4 = ModelTest4()
test4.mainVotingClassifier(data, target, lrP1, lrP2, dTreeP1, dTreeP2, dTreeP3, lsvcP1, unknownStrategy = "most_frequent")
