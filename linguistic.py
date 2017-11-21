import numpy as np
import math


from sklearn import svm
import string
import featureold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectPercentile
from sklearn import tree
from sklearn.linear_model import LogisticRegression


def traintestsplit(fakeFileTrain, realFileTrain,fakeFileTest,realFileTest,leafsize=5,bag=10):
    realTrain,r1 = featureold.constructMat(realFileTrain, 1)
    fakeTrain,r2 = featureold.constructMat(fakeFileTrain, 0)
    realTest,r3 = featureold.constructMat(realFileTest, 1)
    fakeTest,r4 = featureold.constructMat(fakeFileTest, 0)
    dataTrain = np.append(realTrain, fakeTrain, axis=0)
    dataTest = np.append(realTest, fakeTest, axis=0)
    np.random.shuffle(dataTrain)
    np.random.shuffle(dataTest)
        
    X_Train=dataTrain[:,0:-1] # features
    Y_Train=dataTrain[:,-1] # label
    X_Test=dataTest[:,0:-1] #features
    Y_Test=dataTest[:,-1] # label
    return X_Train,Y_Train,X_Test,Y_Test

def trainSVM(x,y):
    clf=svm.SVC(gamma='auto', C=100)
    clf.fit(x,y)
    return clf

def trainRF(x,y):
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(x,y)
    return clf 

def trainLR(x,y):
    clf=LogisticRegression()
    clf.fit(x,y)
    return clf     

def accuracy(predict, test_label):
    acc = 0
   # print(test_label)
    for i in range(len(predict)):
        if predict[i] == test_label[i]:
            acc += 1
    print("acc = ",acc,"/",len(predict))
    return acc*1.0/len(predict)        

if __name__=="__main__":
   
    x_train, y_train, x_test, y_test=traintestsplit(fakeFileTrain='./trainingfake.txt', realFileTrain='./trainingreal.txt',fakeFileTest='./testingfake.txt', realFileTest='./testingreal.txt')

    clf=trainSVM(x_train, y_train)
    print("BY SVM : \n")
    prediction=clf.predict(x_test)
    #print(y_test,"\n")
    #print(prediction)
    print(accuracy(prediction,y_test))

    clfRF=trainRF(x_train, y_train)
    predictionRF=clfRF.predict(x_test)
    print("\n\nBY RANDOM FOREST : \n")
    #print(predictionRF)
    print(accuracy(predictionRF,y_test))
    
    clfLR=trainLR(x_train, y_train)
    predictionLR=clfLR.predict(x_test)
    print("\n\nBY LOGISTIC REGRESSION : \n")
    #print(predictionLR)
    print(accuracy(predictionLR,y_test))
    
    
