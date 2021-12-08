import json
import time
import sys
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession
import numpy as np
import pandas as pd
from textproc import *
import csv


from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.cluster import KMeans

if sys.argv[1] == 'train':
	header = ["SVM_class","NB_Class","RF_class"]
	with open("train_track.csv",'w',newline='') as f:
		writer = csv.writer(f)
		writer.writerow(header)

if sys.argv[1]=='cluster':
	clu_header=['centroid1_diff', 'centroid2_diff']
	with open("cluster_track.csv",'w',newline='') as f1:
		writer = csv.writer(f1)
		writer.writerow(clu_header)

old1=np.array([0,0])
old2=np.array([0,0])

sc = SparkContext.getOrCreate()
sc.setLogLevel('OFF')
ssc = StreamingContext(sc, 1)
spark = SparkSession(sc)

def outr(rdd):
	#print(rdd)
	return json.loads(rdd)

def func(rdd):
	#print(rdd.collect())
	data=rdd.collect()
	if len(data)==0:
		pass
	else:
		df= pd.DataFrame(data[0]).transpose()
		df=preprocess(df)
		#print(df)
		#tfidfconverter = TfidfVectorizer(max_features=1000, min_df=5, max_df=0.7, stop_words=stopwords_english)  
		#x = tfidfconverter.fit_transform(df['feature1']).toarray()
		#y = df['feature0'].apply(lambda x:int(x))
		
		if sys.argv[1]=='train':
			tfidfconverter = TfidfVectorizer(max_features=1000, min_df=5, max_df=0.7, stop_words=stopwords_english)  
			x = tfidfconverter.fit_transform(df['feature1']).toarray()
			y = df['feature0'].apply(lambda x:int(x))
			X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.05, random_state =26105111)
			sc1=svm_classifier(X_train, X_test, y_train, y_test)
			sc2=NB_classifier(X_train, X_test, y_train, y_test)
			sc3=Rf_classifier(X_train, X_test, y_train, y_test)
			print("Score c1: %s  ,Score c2: %s  ,Score: %s  "%(str(sc1),str(sc2),str(sc3)))
			with open("train_track.csv",'a',newline='') as f:
				writer = csv.writer(f)
				data=[sc1,sc2,sc3]
				writer.writerow(data)
		
		elif sys.argv[1]=='cluster':
			global old1
			global old2
			tfidfconverter = TfidfVectorizer(max_features=1000, min_df=5, max_df=0.7, stop_words=stopwords_english)  
			x = tfidfconverter.fit_transform(df['feature1']).toarray()
			y = df['feature0'].apply(lambda x:int(x))
			X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.05, random_state =26105111)
			km = KMeans(n_clusters=2, init='random',n_init=10, max_iter=5, tol=1e-04, random_state=0)
			y_km = km.fit_predict(X_train)
			new1= km.cluster_centers_[:,0]
			new2= km.cluster_centers_[:,1]
			dist1= np.linalg.norm(new1 - old1)
			dist2= np.linalg.norm(new2 - old2)
			data = [dist1,dist2]
			print(data)
			with open("cluster_track.csv",'a',newline='') as f1:
				writer = csv.writer(f1)
				writer.writerow(data)
			
			if min(data) <= 0.0005:
				print("###########  small value of centroid shift detected ########################")
				
			old1=new1
			old2=new2
			
			
			
		else:
			tfidfconverter = TfidfVectorizer(max_features=275, min_df=5, max_df=0.7, stop_words=stopwords_english) 
			X_test= tfidfconverter.fit_transform(df['feature1']).toarray()
			y_test= df['feature0'].apply(lambda x:int(x))
			test_cycle(X_test,y_test)
			
		
	
data = ssc.socketTextStream("localhost", 6100)


j_data= data.map(outr).foreachRDD(func)


ssc.start()
ssc.awaitTermination()
