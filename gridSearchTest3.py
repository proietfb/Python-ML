import numpy as np
import matplotlib as plt
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score

class ModelTest3():

    lrParams = []
    dTreeParams = []
    lsvcParams = []
    
    def __init__(self):
        pass

    def lrGridSearch(self, pipe, data, target, xTrain, yTrain):
    
        paramRange = [0.001,0.01,0.1,1.0,10.0,100.0]

        paramGrid = {'lr__C': paramRange,
                 'lr__solver': ['liblinear','sag']}
    
        gsLR = GridSearchCV(pipe,paramGrid,scoring='accuracy',cv = 10, n_jobs = -1)

        gsLRFit = gsLR.fit(xTrain, yTrain)

        #5x2 cross validation
    
        scores = cross_val_score(gsLR,data,target,scoring='accuracy', cv = 5, n_jobs = -1)
    
        self.lrParams.append(gsLRFit.best_score_)
        self.lrParams.append(gsLRFit.best_params_)
        self.lrParams.append(np.mean(scores))
        self.lrParams.append(np.std(scores))

    def dTreeGridSearch(self,pipe, data, target, xTrain, yTrain):
         
         paramGrid = {'dTree__criterion': ['gini', 'entropy'],
                      'dTree__splitter': ['best','random'],
                      'dTree__max_depth': [1,2,3,4,5,None]}
    
         gsdTree = GridSearchCV(pipe,paramGrid,scoring='accuracy',cv = 10, n_jobs = -1)

         gsdTreeFit = gsdTree.fit(xTrain, yTrain)

         scores = cross_val_score(gsdTree,data,target,scoring='accuracy', cv = 5, n_jobs = -1)
         self.dTreeParams.append(gsdTree.best_score_)
         self.dTreeParams.append(gsdTree.best_params_)
         self.dTreeParams.append(np.mean(scores))
         self.dTreeParams.append(np.std(scores))

    def lsvcGridSearch(self, pipe, data, target, xTrain, yTrain):
    
        paramRange = [0.001,0.01,0.1,1.0,10.0,100.0]
    
        paramGrid = {'lsvc__C': paramRange,
                 'lsvc__penalty':['l2']}
        gsLSVC = GridSearchCV(pipe, paramGrid, scoring='accuracy',cv = 10, n_jobs = -1)

        gsLSVCFit = gsLSVC.fit(xTrain,yTrain)
        scores = cross_val_score(gsLSVC, data, target, scoring='accuracy', cv = 5, n_jobs = -1)
        self.lsvcParams.append(gsLSVCFit.best_score_)
        self.lsvcParams.append(gsLSVCFit.best_params_)
        self.lsvcParams.append(np.mean(scores))
        self.lsvcParams.append(np.std(scores))

    def mainGridSearchCV(self, pipe, data, target, xTrain, yTrain, classifiers, unknownStrategy = 'remove'):

        columns = ["Classifier", "Accuracy (cross_val)", "gridSearch best Params", "removal"]
        
        if(unknownStrategy == "remove"):
            removal = "yes"
        else:
            removal = "no"

        self.lrGridSearch(pipe[0], data, target, xTrain, yTrain)
        print("lrGrid done")
        self.dTreeGridSearch(pipe[1], data, target, xTrain, yTrain)
        print("dTreeGrid done")
        self.lsvcGridSearch(pipe[2], data, target, xTrain, yTrain)
        print("lsvcGrid done")
        
        lr_crossValAcc = ("%.3f" %  self.lrParams[2]) + " +/- " + ("%.3f" %  self.lrParams[3])
        dTree_crossValAcc = ("%.3f" %  self.dTreeParams[2]) + " +/- " + ("%.3f" %  self.dTreeParams[3])
        lsvc_crossValAcc = ("%.3f" %  self.lsvcParams[2]) + " +/- " + ("%.3f" %  self.lsvcParams[3])
        
        data = [[classifiers[0],lr_crossValAcc,self.lrParams[1], removal],
                [classifiers[1],dTree_crossValAcc,self.dTreeParams[1],removal],
                [classifiers[2],lsvc_crossValAcc,self.lsvcParams[1],removal]]

        df = pd.DataFrame(data, columns = columns)

        with open('test3/resultData.csv', 'a') as f:
            df.to_csv(f, sep = ',', index = False)
