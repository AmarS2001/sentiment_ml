import re                                  
import string                             
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer
#from nltk.tokenize import TweetTokenizer  
import pandas as pd
import numpy as np
import string
import pickle



from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import BernoulliNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import csv




stopwords_english = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
stems=PorterStemmer()

#df = pd.read_csv("train.csv").head()
#print(df)

def process1(x):
    x= x.lower()
    x = re.sub(r'^RT[\s]+', '.' , x)
    x = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','url',x)
    x = re.sub(r'#', '' , x)
    x = re.sub(r'[0-9]', '.' , x)
    x = re.sub(r'(\\u[0-9A-Fa-f]+)', '.', x)       
    x = re.sub(r'[^\x00-\x7f]' , '.',x)
    x = re.sub('@[^\s]+','atUser',x)
    x = re.sub(r"(\!)\1+", ' multiExclamation', x)
    x = re.sub(r"(\?)\1+", ' multiQuestion', x)
    x = re.sub(r"(\.)\1+", ' multistop', x)
    return x

def tokens(x):
    #tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,reduce_len=True)
    #res = tokenizer.tokenize(x)
    return x.split()

def process2(x):
    res=[]
    for word in x:
        if word not in stopwords_english and word not in string.punctuation:
            res.append(word)
    return res

def process3(x):
    res=[]
    for word in x:
        stem_word= stems.stem(word)
        res.append(stem_word)
    return " ".join(res)
    


def preprocess(df):
    df['feature1']= df['feature1'].apply(lambda x: process1(x))
    df['feature1']= df['feature1'].apply(lambda x: tokens(x))
    df['feature1']= df['feature1'].apply(lambda x: process2(x))
    df['feature1']= df['feature1'].apply(lambda x: process3(x))
    return df



print("####### processed ######")

def svm_classifier(X_train, X_test, y_train, y_test):
	SVCmodel = svm.LinearSVC()
	SVCmodel.fit(X_train, y_train)
	y_pred2 = SVCmodel.predict(X_test)
	return accuracy_score(y_test,y_pred2)

def NB_classifier(X_train, X_test, y_train, y_test):
	clf = RandomForestClassifier(max_depth=2, random_state=0)
	clf.fit(X_train,y_train)
	y_pred2= clf.predict(X_test)
	return accuracy_score(y_test,y_pred2)
	

def Rf_classifier(X_train, X_test, y_train, y_test):
	clf=BernoulliNB()
	clf.fit(X_train,y_train)
	y_pred2=clf.predict(X_test)
	return accuracy_score(y_test,y_pred2)
	


