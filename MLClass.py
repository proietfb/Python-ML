import numpy as np
import operator as op
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

class MLClass():

    def __init__(self):
        pass
    
    def manageNaNValues(self, importFile, names,  strategy = 'most_frequent'):

        if (op.eq(strategy,'remove')):
            importFile = importFile.dropna()
        else:
            for name in names[:-1]:
                if(importFile[name].isnull().sum() > 0):
                    print(name, importFile[name].isnull().sum())
                    mostFrequent = importFile[name].value_counts().idxmax()
                    importFile[name] = importFile[name].fillna(mostFrequent)                
        return importFile

    def checkDiscreteCategories(self, importFile, names): #check which categories are discrete and which continuos
        categoricalFeatures = []
        for name in names:
            if(op.eq(importFile[name].dtypes, np.object)):
                categoricalFeatures.append(True)
            elif(op.eq(importFile[name].dtypes, np.int64)):
                categoricalFeatures.append(False)
        return categoricalFeatures

    def labelEncodingProcess(self, importFile, names): #preprocessing dataset: define discrete string columns as integer labels
        colToEdit = self.checkDiscreteCategories(importFile, names)
        editFile = []
        valuesTransposed = importFile.transpose()
        i = 0
        for row in valuesTransposed.itertuples():
            if(op.eq(colToEdit[i], True)):
                le = LabelEncoder()
                le.fit(row)
                editFile.append(le.transform(row[1:]))
                i+=1
            else:
                editFile.append(row[1:])
                i+=1
        df = pd.DataFrame(data = editFile)
        df = df.transpose()
        df.columns = [names]
        retValues = [df,colToEdit]
        return retValues

    def plotLearningCurve(self, trainSizes, trainMean, testMean, trainStd, testStd, title):
        plt.plot(trainSizes, trainMean, color = 'blue', marker= 'o', markersize = 5, label = 'training accuracy')
        plt.fill_between(trainSizes, trainMean+trainStd, trainMean-trainStd,alpha=0.15, color='blue')
        plt.plot(trainSizes, testMean, color = 'green', linestyle='--', marker='s', markersize=5, label = 'validation accuracy')
        plt.fill_between(trainSizes, testMean+testStd, testMean-testStd, alpha= 0.15, color = 'green')
        plt.grid()
        plt.xlabel('# of training samples')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title(title)
        plt.ylim([0.6, 1.0])
        plt.show()

    def pipePreprocessing(self, trainFrame, testFrame, names,unknownValue = ' ?', unknownStrategy = 'remove'):

        frame = trainFrame.append(testFrame, ignore_index = True)
        frame = frame.replace(unknownValue, np.nan)
        frame = self.manageNaNValues(frame, names, unknownStrategy)
        retValues = self.labelEncodingProcess(frame, names)
        frame = retValues[0]
        categoricalFeatures = retValues[1]

        return frame
        
