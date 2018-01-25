import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

class RandomModelTest1():

    lrScores = []
    dTreeScores = []
    lsvcScores = []

    lrTrainScores = []
    dTreeTrainScores = []
    lsvcTrainScores = []
    
    def __init(self):
        pass

    def randomModel(self,xTrain, yTrain, xTest, yTest):
        


        for run in range(10):
            lr = LogisticRegression()
            lr = lr.fit(xTrain, yTrain)
            self.lrTrainScores.append(lr.score(xTrain, yTrain))
            lrPred = lr.predict(xTest)
            lrScore = accuracy_score(yTest, lrPred)
            self.lrScores.append(lrScore)
            lr = 0
            
            dTree = DecisionTreeClassifier()
            dTree = dTree.fit(xTrain, yTrain)
            self.dTreeTrainScores.append(dTree.score(xTrain, yTrain))
            dTree = dTree.predict(xTest)
            dTreeScore = accuracy_score(yTest, dTree)
            self.dTreeScores.append(dTreeScore)
            dTree = 0
            
            lsvc = LinearSVC()
            lsvc = lsvc.fit(xTrain, yTrain)
            self.lsvcTrainScores.append(lsvc.score(xTrain, yTrain))
            lsvc = lsvc.predict(xTest)
            lsvcScore = accuracy_score(yTest, lsvc)
            self.lsvcScores.append(lsvcScore)
            lsvc = 0

        plt.plot(range(0,10,1), self.lrScores, color = 'green', label = 'Logistic Regression')
        plt.plot(range(0,10,1), self.dTreeScores, color = 'blue', label = 'Decision Tree')
        plt.plot(range(0,10,1), self.lsvcScores, color = 'red', label = 'Linear SVC')
        plt.grid()
        plt.xlabel('run #')
        plt.ylabel('mean score accuracy')
        plt.ylim([0.0,1.0])
        plt.title("Random Model")
        plt.legend()
        plt.show()
        
    def mainRandomModel(self , xTrain, yTrain, xTest, yTest, classifiers, unknownStrategy = "remove"):

        self.randomModel(xTrain,yTrain,xTest, yTest)
        
        lrScoresMean = np.mean(self.lrScores)
        lrScoresSTD = np.std(self.lrScores)
        dTreeScoresMean = np.mean(self.dTreeScores)
        dTreeScoresSTD = np.std(self.dTreeScores)
        lsvcScoresMean = np.mean(self.lsvcScores)
        lsvcScoresSTD = np.std(self.lsvcScores)

        lrTrainScoresMean = np.mean(self.lrTrainScores)
        lrTrainScoresSTD = np.std(self.lrTrainScores)
        dTreeTrainScoresMean = np.mean(self.dTreeTrainScores)
        dTreeTrainScoresSTD = np.std(self.dTreeTrainScores)
        lsvcTrainScoresMean = np.mean(self.lsvcTrainScores)
        lsvcTrainScoresSTD = np.std(self.lsvcTrainScores)

        lrAcc = ("%.3f" % lrScoresMean) + " +/- " + ("%.3f" % lrScoresSTD)
        dTreeAcc = ("%.3f" % dTreeScoresMean) + " +/- " + ("%.3f" % dTreeScoresSTD)
        lsvcAcc = ("%.3f" % lsvcScoresMean) + " +/- " + ("%.3f" % lsvcScoresSTD)

        lrTrainAcc = ("%.3f" % lrTrainScoresMean) + " +/- " + ("%.3f" % lrTrainScoresSTD)
        dTreeTrainAcc = ("%.3f" % dTreeTrainScoresMean) + " +/- " + ("%.3f" % dTreeTrainScoresSTD)
        lsvcTrainAcc = ("%.3f" % lsvcTrainScoresMean) + " +/- " + ("%.3f" % lsvcTrainScoresSTD)
        

        if(unknownStrategy == "remove"):
            removal = "yes"
        else:
            removal = "no"

        columns = ["Classifier", "Train Accuracy (Mean)","Accuracy (mean)", "removal unknown"]

        data = [[classifiers[0],lrTrainAcc, lrAcc, removal],
                [classifiers[1],dTreeTrainAcc, dTreeAcc, removal],
                [classifiers[2],lsvcTrainAcc, lsvcAcc, removal]]

        df = pd.DataFrame(data,columns = columns)

        with open('test1/resultData.csv', 'a') as f:
            df.to_csv(f, sep = ',', index = False)


        

        
        
