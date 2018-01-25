import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.model_selection import cross_val_score, learning_curve

from MLClass import MLClass

class ModelTest2(MLClass):

    lrTrainTestMeanSTD = []
    dTreeTrainTestMeanSTD = []
    lsvcTrainTestMeanSTD = []

    def __init__(self):
        MLClass.__init__(self)

    def lrLearningCurve(self, pipe, data, target):
        lrTrainSize, lrTrainScores, lrTestScores = learning_curve(pipe, data, target, cv = 10, n_jobs = -1)
            
        lrTrainMean = np.mean(lrTrainScores, axis = 1)
        lrTrainSTD = np.std(lrTrainScores, axis = 1)
        lrTestMean = np.mean(lrTestScores, axis = 1)
        lrTestSTD = np.std(lrTestScores, axis = 1)

        self.plotLearningCurve(lrTrainSize, lrTrainMean, lrTestMean, lrTrainSTD, lrTestSTD, "Learning curve Logistic regression")
        self.lrTrainTestMeanSTD = [lrTrainMean.mean(), lrTrainSTD.std(), lrTestMean.mean(), lrTestSTD.std()]


    def dTreeLearningCurve(self, pipe, data, target):
     
        dTreeTrainSize, dTreeTrainScores, dTreeTestScores = learning_curve(pipe, data, target, cv = 10, n_jobs = -1)
        
        dTreeTrainMean = np.mean(dTreeTrainScores, axis = 1)
        dTreeTrainSTD = np.std(dTreeTrainScores, axis = 1)
        dTreeTestMean = np.mean(dTreeTestScores, axis = 1)
        dTreeTestSTD = np.std(dTreeTestScores, axis = 1)

        self.plotLearningCurve(dTreeTrainSize, dTreeTrainMean,dTreeTestMean, dTreeTrainSTD, dTreeTestSTD,"Learning curve Decision Tree Classifier")
        self.dTreeTrainTestMeanSTD = [dTreeTrainMean.mean(), dTreeTrainSTD.std(), dTreeTestMean.mean(), dTreeTestSTD.std()]
    
    def lsvcLearningCurve(self, pipe, data, target):

        lsvcTrainSize, lsvcTrainScores, lsvcTestScores = learning_curve(pipe, data, target, cv = 10, n_jobs = -1)

        lsvcTrainMean = np.mean(lsvcTrainScores, axis = 1)
        lsvcTrainSTD = np.std(lsvcTrainScores, axis = 1)
        lsvcTestMean = np.mean(lsvcTestScores, axis = 1)
        lsvcTestSTD = np.std(lsvcTestScores, axis = 1)

        self.plotLearningCurve(lsvcTrainSize, lsvcTrainMean, lsvcTestMean, lsvcTrainSTD, lsvcTestSTD, "Learning curve Linear SVC")

        self.lsvcTrainTestMeanSTD = [lsvcTrainMean.mean(), lsvcTrainSTD.std(), lsvcTestMean.mean(), lsvcTestSTD.std()]
        
    def mainPipeModel(self, pipe, data, target, classifiers, unknownStrategy = 'remove'):

        columns = ["Classifier", "Train Accuracy (Mean)", "Test Accuracy (Mean)", "preprocessing", "removal"]
        
        if(unknownStrategy == "remove"):
            removal = "yes"
        else:
            removal = "no"

       
        
        self.lrLearningCurve(pipe[0], data, target)

        lrAccTrain =("%.3f" % self.lrTrainTestMeanSTD[0]) + " +/- " + ("%.3f" % self.lrTrainTestMeanSTD[1])
        lrAccTest = ("%.3f" % self.lrTrainTestMeanSTD[2]) + " +/- " + ("%.3f" % self.lrTrainTestMeanSTD[3])
        
        self.dTreeLearningCurve(pipe[1], data, target)

        dTreeAccTrain = ("%.3f" % self.dTreeTrainTestMeanSTD[0]) + " +/- " + ("%.3f" % self.dTreeTrainTestMeanSTD[1])
        dTreeAccTest = ("%.3f" % self.dTreeTrainTestMeanSTD[2]) + " +/- " + ("%.3f" % self.dTreeTrainTestMeanSTD[3])
        
        self.lsvcLearningCurve(pipe[2], data, target)

        lsvcAccTrain = ("%.3f" % self.lsvcTrainTestMeanSTD[0]) + " +/- " + ("%.3f" % self.lsvcTrainTestMeanSTD[1])
        lsvcAccTest = ("%.3f" % self.lsvcTrainTestMeanSTD[2]) + " +/- " + ("%.3f" % self.lsvcTrainTestMeanSTD[3])

        data =[[classifiers[0],lrAccTrain,lrAccTest, "StandardScaler, PCA", removal],
               [classifiers[1],dTreeAccTrain,dTreeAccTest, "regularization 'l2'", removal],
               [classifiers[2],lsvcAccTrain,lsvcAccTest, "StandardScaler, PCA", removal]]

        df = pd.DataFrame(data, columns = columns)

        with open('test2/resultData.csv', 'a') as f:
            df.to_csv(f,sep = ',',index = False)
               
