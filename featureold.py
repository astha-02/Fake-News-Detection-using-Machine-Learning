"""
    function:
    k-fold cross validation
    """
__author__ = "Shiyi Li"

# sentence length(# of char)
# sentence length(# of words)
# # of punctuations
# # of illegal punctuations
import string
import numpy as np
import re
import nltk
#nltk.download()
from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def count_2grams(raw):
    s=[]
    tokens = nltk.word_tokenize(raw)
    tags = nltk.pos_tag(tokens)
    nouns = [word for word,pos in tags if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS')]
    bigram=ngrams(nouns,2)
    s=list(bigram)
    
    return str(s)

def count_1grams(raw):
    s=[]
    tokens = nltk.word_tokenize(raw)
    tags = nltk.pos_tag(tokens)
    nouns = [word for word,pos in tags if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS')]
    unigram=ngrams(nouns,1)
    s=list(unigram)
    
    return str(s)
 
def count_3grams(raw):
    s=[]
    tokens = nltk.word_tokenize(raw)
    tags = nltk.pos_tag(tokens)
    nouns = [word for word,pos in tags if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS')]
    trigram=ngrams(nouns,3)
    s=list(trigram)
    return str(s)

def freq2(raw):
    string1=count_2grams(raw);
    list1=string1.split(' ')
    freq=0
    for grams in list1:
       freq +=1
    return freq

def freq1(raw):
    string1=count_1grams(raw);
    list1=string1.split(' ')
    freq=0
    for grams in list1:
       freq +=1
    return freq

def freq3(raw):
    string1=count_3grams(raw);
    list1=string1.split(' ')
    freq=0
    for grams in list1:
       freq +=1
    return freq

def tfidf(corpus):
    
    tfidf_vect=TfidfVectorizer(min_df=1, stop_words='english')
    tfidf_matrix=tfidf_vect.fit_transform(corpus)
    print(tfidf_matrix.todense())	

def extract_adjective(sentences):
    adj_sentences = list()
    count=0;
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        adj_tags = nltk.pos_tag(words)
        one_adj_sentence = ""
        for index, tag in enumerate(adj_tags, start = 0):
            one_tag = tag[1]
            if one_tag in ['JJ', 'JJR', 'JJS']:
                one_adj_sentence += words[index]
                one_adj_sentence += " "
                count+=1
        adj_sentences.append(one_adj_sentence)
        #print(one_adj_sentence)
    #return adj_sentences
    return count

def removePunc(input):
    '''
    :param input: string
    :return: string, without the punctuations
    '''
    #return input.translate(string.maketrans("",""), string.punctuation)
    return re.sub("[\.\t\,\:;\(\)\.]", "", input, 0, 0)

def numOfWords(input):
    '''
    :param input: string
    :return: number of words, number of continuous space
    '''
    splitted = input.split(" ")
    res=0
    for i in splitted:
        if len(i)>0:
            res+=1
    return res

def numOfChar(input):
    '''
    :param input: string
    :return: number of char
    '''
    return len(input)

def numOfPunc(input):
    '''
    :param input: string
    :return: number of punctuations
    '''
    return len(input)-len(removePunc(input))

def numOfContPunc(input):
    res=0;
    state=False
    for i in range(1,len(input)):
        if input[i] in string.punctuation:
            if input[i-1] in string.punctuation:
                if state:
                    pass
                else:
                    state=True
                    res+=1
            else:
                state=False
                pass
        else:
            state=False
    return res

def numOfContUpperCase(input):
    res = 0;
    state = False
    for i in range(1, len(input)):
        if input[i].isupper():
            if input[i - 1].isupper():
                if state:
                    pass
                else:
                    state = True
                    res += 1
            else:
                state = False
                pass
        else:
            state = False
    return res
    pass

def constructMat(file,label):
    '''
    :param file: input file
    :param label: the label of the data in the file
    :return: ndarray
    '''
    res=np.array([])
    line1=True
    with open(file) as data:
        for line in data:
            if line1:
                line1=False
                cleaned = line.lower().strip()
                original = line.strip()
                fea1 = numOfWords(cleaned)
                fea2 = numOfChar(cleaned)
                fea3 = numOfPunc(cleaned)
                fea4 = numOfContPunc(cleaned)
                fea5 = numOfContUpperCase(original)
                fea6 = extract_adjective(cleaned)
                fea7 = count_2grams(cleaned)
                fea8 = freq2(cleaned) 
                fea9 = count_1grams(cleaned)
                fea10 = freq1(cleaned)
                fea11 = count_3grams(cleaned)
                fea12 = freq3(cleaned)
                #fea13 = tfidf(fea7)
                #res1= np.array([[line,fea1, fea2, fea3, fea4, fea5, fea6,label,count_1grams(),count_2grams(),count_3grams()]])
                res1= np.array([[line,fea1, fea2, fea3, fea4, fea5, fea6,fea7,fea8,fea9,fea10,fea11,fea12,label]])
                res = np.array([[fea1, fea2, fea3, fea4, fea5, label]])
            else:
                cleaned = line.lower().strip()
                original = line.strip()
                fea1 = numOfWords(cleaned)
                fea2 = numOfChar(cleaned)
                fea3 = numOfPunc(cleaned)
                fea4 = numOfContPunc(cleaned)
                fea5 = numOfContUpperCase(original)
                fea6 = extract_adjective(cleaned)
                fea7 = count_2grams(cleaned)
                fea8 = freq2(cleaned)
                fea9 = count_1grams(cleaned)
                fea10 = freq1(cleaned)
                fea11 = count_3grams(cleaned)
                fea12 = freq3(cleaned)
                #fea13 = tfidf(fea7)
                #newrow1= np.array([[line,fea1, fea2, fea3, fea4, fea5, fea6,label,count_1grams(),count_2grams(),count_3grams()]])
                newrow1= np.array([[line,fea1, fea2, fea3, fea4, fea5, fea6,fea7,fea8,fea9,fea10,fea11,fea12,label]])
                newrow = np.array([[fea1, fea2, fea3, fea4, fea5, label]])
                res = np.append(res, newrow, axis=0)
                res1= np.append(res1, newrow1, axis=0)
    
    
    #print(my_df)
    return res,res1

def constructRealFea(headline):
    cleaned = headline.lower().strip()
    original = headline.strip()
    fea1 = numOfWords(cleaned)
    fea2 = numOfChar(cleaned)
    fea3 = numOfPunc(cleaned)
    fea4 = numOfContPunc(cleaned)
    fea5 = numOfContUpperCase(original)
    res = np.array([[fea1, fea2, fea3, fea4, fea5]])
   
    
    return res


if __name__ == '__main__':
    #print numOfContUpperCase("huhAAiAihiuhAAAAuhuhAAAAA")
    resFake,res1=constructMat('./fake2.txt',1)
    resReal,res2=constructMat('./real2.txt',0)
    
    

    my_df=pd.DataFrame(res1)
    df=pd.DataFrame(res2)
    df_merge=pd.concat([my_df,df],ignore_index=True)
    #df_merge=df_merge.sample(frac=1).reset_index(drop=True)
    #my_df.to_csv('extracted1.csv',index=False,header=('Headline','NumOfWords', 'NumOfChar',   'NumOfPunc','NumOfContPunc','NumOfContUpperCase','NumOfAdjectives','Label','Unigrams','Bigrams','Trigrams'))
    df_merge.to_csv('extracted1.csv',index=False,header=('Headline','NumOfWords', 'NumOfChar',   'NumOfPunc','NumOfContPunc','NumOfContUpperCase','NumOfAdjectives','Bigrams','Bi-freq','unigrams','un-freq','trigram','tri-freq','Label'))
