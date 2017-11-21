import numpy as np
import pandas as pd
import nltk
import string
from nltk import word_tokenize
from nltk.util import ngrams
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from scipy.sparse import coo_matrix, hstack
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

def get_tokens(training_data):
    tokens = []
    for sentence in training_data:
        for word in sentence.split(' '):
            lowers = word.lower()
            lowers = lowers.replace('\n','')
            for punc in string.punctuation:
                lowers = lowers.replace(punc, '')
            for num in "0123456789":
                lowers = lowers.replace(num, "")
            if lowers != " " and lowers != "" and lowers not in stopwords.words('english'):
                if is_ascii(lowers): 
                    tokens.append(str(lowers))
            
    return tokens

def get_shallow_POS(training_data):
    counter_list = []
    for sentence in training_data:
        counter_list.append(Counter([k if k not in string.punctuation else "PUNCT" for k in [j for i,j in pos_tag(word_tokenize(sentence))]]))

    return sum(counter_list, Counter())

def is_ascii(s):
    return all(ord(c) < 128 for c in s)

def get_classification_accuracy(predict, test_label):
    TP = 0
    TN = 0
    POS = 0
    NEG = 0
    
    for i in range(len(predict)):
        if predict[i] == test_label[i] and predict[i] == 1:
            TP += 1
        elif predict[i] == test_label[i] and predict[i] == 0:
            TN += 1
        if predict[i] == 1:
            POS += 1
        if predict[i] == 0:
            NEG += 1
            
    TPR = TP*1.0/(POS)
    TNR = TN*1.0/(NEG)
    
    class_accuracy = (TP+TN)*1.0/(len(predict))
    return (TPR, TNR, class_accuracy)

def train_RF(ngram_vect, tokens_counter, tfidf, pos_vect, pos_counter, training_data, training_label):
    ngram_train = ngram_vect.fit(Counter(tokens_counter)).transform(training_data)
    tfidf_train = tfidf.fit(Counter(tokens_counter)).transform(training_data)
    X_train = hstack([ngram_train, tfidf_train])
    pos_train = pos_vect.fit(pos_counter).transform(training_data)
    X_train = hstack([X_train, pos_train])

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, training_label)
    return clf

def test_RF(ngram_vect, tfidf, pos_vect, clf2, test_data):
    
    ngram_test = ngram_vect.transform(test_data)
    tfidf_test = tfidf.transform(test_data)
    X_test = hstack([ngram_test, tfidf_test])
    pos_test = pos_vect.transform(test_data)
    X_test = hstack([X_test, pos_test])
    predict_test = clf2.predict(X_test)

    return predict_test

def train_SVM(ngram_vect, tokens_counter, tfidf, pos_vect, pos_counter, training_data, training_label):
    
    ngram_train = ngram_vect.fit(Counter(tokens_counter)).transform(training_data)
    tfidf_train = tfidf.fit(Counter(tokens_counter)).transform(training_data)
    X_train = hstack([ngram_train, tfidf_train])
    pos_train = pos_vect.fit(pos_counter).transform(training_data)
    X_train = hstack([X_train, pos_train])
    
    svm_instance = svm.SVC(gamma='auto', C=100)
    svm_instance.fit(X_train, training_label)
    return svm_instance

    

def test_SVM(ngram_vect, tfidf, pos_vect, clf, test_data):
    
    ngram_test = ngram_vect.transform(test_data)
    tfidf_test = tfidf.transform(test_data)
    X_test = hstack([ngram_test, tfidf_test])
    pos_test = pos_vect.transform(test_data)
    X_test = hstack([X_test, pos_test])
    predict_test = clf.predict(X_test)

    return predict_test

def train_LR(ngram_vect,tokens_counter, tfidf, pos_vect, pos_counter, training_data, training_label):

    ngram_train = ngram_vect.fit(Counter(tokens_counter)).transform(training_data)
    tfidf_train = tfidf.fit(Counter(tokens_counter)).transform(training_data)
    X_train = hstack([ngram_train, tfidf_train])
    pos_train = pos_vect.fit(pos_counter).transform(training_data)
    X_train = hstack([tfidf_train, pos_train])
    logistic_regression_model = LogisticRegression()
    logistic_regression_model.fit(X_train, training_label)
    return logistic_regression_model

def test_LR(ngram_vect,tfidf, pos_vect, clf1, test_data):

    ngram_test = ngram_vect.transform(test_data)
    tfidf_test = tfidf.transform(test_data)
    X_test = hstack([ngram_test, tfidf_test])
    pos_test = pos_vect.transform(test_data)
    X_test = hstack([tfidf_test, pos_test])
    predict_test = clf1.predict(X_test)

    return predict_test

def trainSVMonUnigram(ngram_vect, tokens_counter,training_data,training_label):
    X_train = ngram_vect.fit(Counter(tokens_counter)).transform(training_data)
    svm_instance = svm.SVC(gamma='auto', C=100)
    svm_instance.fit(X_train, training_label)
    return svm_instance

def testSVMonUnigram(ngram_vect,clf,test_data):
     X_test = ngram_vect.transform(test_data)
     predict_test = clf.predict(X_test)
     return predict_test

def trainSVMonNgram(vectorizer,training_data,training_label):
    X_train=vectorizer.fit_transform(training_data)
    svm_instance = svm.SVC(gamma='auto', C=100)
    svm_instance.fit(X_train, training_label)
    return svm_instance

def testSVMonNgram(vectorizer,clfbg,test_data):
     X_test = vectorizer.transform(test_data)
     predict_test = clfbg.predict(X_test)
     return predict_test

def trainRFonNgram(vectorizer,training_data,training_label):
    X_train=vectorizer.fit_transform(training_data)
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, training_label)
    return clf

def testRFonNgram(vectorizer,clfbg,test_data):
     X_test = vectorizer.transform(test_data)
     predict_test = clfbg.predict(X_test)
     return predict_test

def trainLRonNgram(vectorizer,training_data,training_label):
    X_train=vectorizer.fit_transform(training_data)
    clf = LogisticRegression()
    clf.fit(X_train, training_label)
    return clf

def testLRonNgram(vectorizer,clfbg,test_data):
     X_test = vectorizer.transform(test_data)
     predict_test = clfbg.predict(X_test)
     return predict_test

    
def model_accuracy(trained_model, features, targets):
    """
    Get the accuracy score of the model
    :param trained_model:
    :param features:
    :param targets:
    :return:
    """
    accuracy_score = trained_model.score(features, targets)
    return accuracy_score

def accuracy(predict, test_label):
    acc = 0
   # print(test_label)
    for i in range(len(predict)):
        if predict[i] == test_label[i]:
            acc += 1
    print("acc = ",acc,"/",len(predict))
    return acc*1.0/len(predict)

def count01(data):
    count_1 = 0
    count_0 = 0

    for i in data:
        if i == 1:
            count_1 += 1
        else:
            count_0 += 1
    return count_1,count_0
    


if __name__ == '__main__':

    df=pd.read_csv('newfile1.csv',names=['Headline','Label'])
    df.loc[df["Label"]=='1',"Label",]=1
    df.loc[df["Label"]=='0',"Label",]=0
    #print(df.head())
    df_x=df["Headline"]
    df_y=df["Label"]
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.60,random_state=5)
    y_train=pd.factorize(y_train)[0]
    y_test=pd.factorize(y_test)[0]
    #print(y_test)
    tokens_counter = get_tokens(x_train)
    pos_counter = get_shallow_POS(x_train)
    ngram_vect = CountVectorizer(min_df=1)
    tfidf = TfidfVectorizer(min_df=1)
    pos_vect = CountVectorizer(min_df=1)
    clf = train_SVM(ngram_vect, tokens_counter, tfidf, pos_vect, pos_counter, x_train, y_train)
    clf1 = train_LR(ngram_vect,tokens_counter, tfidf, pos_vect, pos_counter, x_train, y_train)
    clf2 = train_RF(ngram_vect,tokens_counter, tfidf, pos_vect, pos_counter, x_train, y_train) 

    prediction = test_SVM(ngram_vect, tfidf, pos_vect, clf, x_test)
    
    print("..................Test_accuracy_words+pos+tfidf......................")
    acc = accuracy(prediction,y_test)
    
    predict1 = test_RF(ngram_vect,tfidf, pos_vect, clf2, x_test)
    
    test_accuracy_RF = accuracy(predict1, y_test)

    predict = test_LR(ngram_vect,tfidf, pos_vect, clf1, x_test)
    
    test_accuracy = accuracy(predict, y_test)
    
    

    print("test_accuracy : SVM ",acc)
    print("test_accuracy : RF = ",test_accuracy_RF)
    print("test_accuracy : LR = ",test_accuracy)
    
    print("\n.........SUPPORT VECTOR MACHINE........... \n")    

    svmuni=trainSVMonUnigram(ngram_vect, tokens_counter,x_train,y_train)
    predictuni=testSVMonUnigram(ngram_vect,svmuni,x_test)
    #print("prediction by SVM on Unigram = \n",predictuni)
    test_accuracy_unisvm = accuracy(predictuni, y_test)
    print("test_accuracy_unisvm :  = ",test_accuracy_unisvm)
    
    for i in range(2,4,1):
        vectorizer=CountVectorizer(ngram_range=(i,i),token_pattern=r'\b\w+\b',stop_words='english',min_df=1)
        clfbg=trainSVMonNgram(vectorizer,x_train,y_train)
        predictbg=testSVMonNgram(vectorizer,clfbg,x_test)
        #print("prediction by SVM on",i,"gram = \n",predictbg)
        test_accuracy_bgsvm = accuracy(predictbg, y_test)
        print("test_accuracy_",i,"gramsvm :  = ",test_accuracy_bgsvm)
    
    
    vectorizer=CountVectorizer(ngram_range=(1,2),token_pattern=r'\b\w+\b',min_df=1)
    clfbg=trainSVMonNgram(vectorizer,x_train,y_train)
    predictbg=testSVMonNgram(vectorizer,clfbg,x_test)
    #print("prediction by SVM on uni+bigram = \n",predictbg)
    test_accuracy_bgsvm = accuracy(predictbg, y_test)
    print("test_accuracy_uni+bigram :  = ",test_accuracy_bgsvm)

    vectorizer=CountVectorizer(ngram_range=(2,3),token_pattern=r'\b\w+\b',min_df=1)
    clfbg=trainSVMonNgram(vectorizer,x_train,y_train)
    predictbg=testSVMonNgram(vectorizer,clfbg,x_test)
    #print("prediction by SVM on uni+bigram = \n",predictbg)
    test_accuracy_bgsvm = accuracy(predictbg, y_test)
    print("test_accuracy_bigram+tri :  = ",test_accuracy_bgsvm)

    vectorizer=CountVectorizer(ngram_range=(1,3),token_pattern=r'\b\w+\b',min_df=1)
    clfbg=trainSVMonNgram(vectorizer,x_train,y_train)
    predictbg=testSVMonNgram(vectorizer,clfbg,x_test)
    #print("prediction by SVM on uni+bigram = \n",predictbg)
    test_accuracy_bgsvm = accuracy(predictbg, y_test)
    print("test_accuracy_uni+bigram+trigram :  = ",test_accuracy_bgsvm)

################################################################################
    print("\n.........RANDOM FOREST........... \n")
    for i in range(1,4,1):
        vectorizer=CountVectorizer(ngram_range=(i,i),token_pattern=r'\b\w+\b',min_df=1)
        clfbg=trainRFonNgram(vectorizer,x_train,y_train)
        predictbg=testRFonNgram(vectorizer,clfbg,x_test)
        #print("prediction by RF on",i,"gram = \n",predictbg)
        test_accuracy_bgsvm = accuracy(predictbg, y_test)
        print("test_accuracy_",i,"gramrf :  = ",test_accuracy_bgsvm)

    vectorizer=CountVectorizer(ngram_range=(1,2),token_pattern=r'\b\w+\b',min_df=1)
    clfbg=trainRFonNgram(vectorizer,x_train,y_train)
    predictbg=testRFonNgram(vectorizer,clfbg,x_test)
    #print("prediction by SVM on uni+bigram = \n",predictbg)
    test_accuracy_bgsvm = accuracy(predictbg, y_test)
    print("test_accuracy_uni+bigram :  = ",test_accuracy_bgsvm)

    vectorizer=CountVectorizer(ngram_range=(2,3),token_pattern=r'\b\w+\b',min_df=1)
    clfbg=trainRFonNgram(vectorizer,x_train,y_train)
    predictbg=testRFonNgram(vectorizer,clfbg,x_test)
    #print("prediction by SVM on uni+bigram = \n",predictbg)
    test_accuracy_bgsvm = accuracy(predictbg, y_test)
    print("test_accuracy_bigram+tri :  = ",test_accuracy_bgsvm)

    vectorizer=CountVectorizer(ngram_range=(1,3),token_pattern=r'\b\w+\b',min_df=1)
    clfbg=trainRFonNgram(vectorizer,x_train,y_train)
    predictbg=testRFonNgram(vectorizer,clfbg,x_test)
    #print("prediction by SVM on uni+bigram = \n",predictbg)
    test_accuracy_bgsvm = accuracy(predictbg, y_test)
    print("test_accuracy_uni+bigram+trigram :  = ",test_accuracy_bgsvm)
################################################################################

    print("\n.........LOGISTIC REGRESSION........... \n")
    for i in range(1,4,1):
        vectorizer=CountVectorizer(ngram_range=(i,i),token_pattern=r'\b\w+\b',min_df=1)
        clfbg=trainLRonNgram(vectorizer,x_train,y_train)
        predictbg=testLRonNgram(vectorizer,clfbg,x_test)
        #print("prediction by RF on",i,"gram = \n",predictbg)
        test_accuracy_bgsvm = accuracy(predictbg, y_test)
        print("test_accuracy_",i,"gramLR :  = ",test_accuracy_bgsvm)
    
    vectorizer=CountVectorizer(ngram_range=(1,2),token_pattern=r'\b\w+\b',min_df=1)
    clfbg=trainLRonNgram(vectorizer,x_train,y_train)
    predictbg=testLRonNgram(vectorizer,clfbg,x_test)
    #print("prediction by SVM on uni+bigram = \n",predictbg)
    test_accuracy_bgsvm = accuracy(predictbg, y_test)
    print("test_accuracy_uni+bigram :  = ",test_accuracy_bgsvm)

    vectorizer=CountVectorizer(ngram_range=(2,3),token_pattern=r'\b\w+\b',min_df=1)
    clfbg=trainLRonNgram(vectorizer,x_train,y_train)
    predictbg=testLRonNgram(vectorizer,clfbg,x_test)
    #print("prediction by SVM on uni+bigram = \n",predictbg)
    test_accuracy_bgsvm = accuracy(predictbg, y_test)
    print("test_accuracy_bigram+tri :  = ",test_accuracy_bgsvm)

    vectorizer=CountVectorizer(ngram_range=(1,3),token_pattern=r'\b\w+\b',min_df=1)
    clfbg=trainLRonNgram(vectorizer,x_train,y_train)
    predictbg=testLRonNgram(vectorizer,clfbg,x_test)
    #print("prediction by SVM on uni+bigram = \n",predictbg)
    test_accuracy_bgsvm = accuracy(predictbg, y_test)
    print("test_accuracy_uni+bigram+trigram :  = ",test_accuracy_bgsvm)
    

    










