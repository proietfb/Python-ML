import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

class ModelTest4():

    votingScores = []
    
    def __init__(self):
        pass
    
    def votingClassifier(self, data, target, lrP1, lrP2, dTreeP1, dTreeP2, dTreeP3, lsvcP1):
        
        pipeLR = Pipeline([('stds', StandardScaler()),
                   ('pca', PCA(n_components = 2)),
                   ('lr', LogisticRegression(C = lrP1, solver = lrP2))])
        pipeDTree = Pipeline([('feature_selection', SelectFromModel(LinearSVC(penalty = 'l2', dual = False))),
                      ('dTree', DecisionTreeClassifier(criterion = dTreeP1, splitter = dTreeP2, max_depth = dTreeP3))])
        pipeLSVC = Pipeline([('stds', StandardScaler()),
                     ('pca', PCA(n_components = 2)),
                     ('lsvc', LinearSVC(C = lrP1, penalty = 'l2'))])
        
        vC = VotingClassifier([('lr',pipeLR),('dTree',pipeDTree),('lsvc',pipeLSVC)])

        scores = cross_val_score(vC, data, target, cv = 10, scoring = 'accuracy')
    
        self.votingScores.append(np.mean(scores))
        self.votingScores.append(np.std(scores))

    def mainVotingClassifier(self, data, target, lrP1, lrP2, dTreeP1, dTreeP2, dTreeP3, lsvcP1, unknownStrategy = 'remove'):

        columns = ["Classifier", "Accuracy (crossVal)", "removal"]
        
        if(unknownStrategy == "remove"):
            removal = "yes"
        else:
            removal = "no"

        print("starting voting score")
        self.votingClassifier(data, target, lrP1, lrP2, dTreeP1, dTreeP2, dTreeP3, lsvcP1)

        
        accuracy = ("%.3f" % self.votingScores[0]) + " +/- " + ("%.3f" % self.votingScores[1])

        data = [["Voting Classifier", accuracy, removal]]

        df = pd.DataFrame(data, columns = columns)

        with open('test4/resultData.csv', 'a') as f:
            df.to_csv(f, sep=',',index = False)

        
