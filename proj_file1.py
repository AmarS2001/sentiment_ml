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

header = ["SVM_class","NB_Class","RF_class"]
with open("train_track.csv",'w',newline='') as f:
	writer = csv.writer(f)
	writer.writerow(header)

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
		tfidfconverter = TfidfVectorizer(max_features=1000, min_df=5, max_df=0.7, stop_words=stopwords_english)  
		x = tfidfconverter.fit_transform(df['feature1']).toarray()
		y = df['feature0'].apply(lambda x:int(x))
		
		if sys.argv[1]=='train':
			X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.05, random_state =26105111)
			sc1=svm_classifier(X_train, X_test, y_train, y_test)
			sc2=NB_classifier(X_train, X_test, y_train, y_test)
			sc3=Rf_classifier(X_train, X_test, y_train, y_test)
			print("Score c1: %s  ,Score c2: %s  ,Score: %s  "%(str(sc1),str(sc2),str(sc3)))
			with open("train_track.csv",'a',newline='') as f:
				writer = csv.writer(f)
				data=[sc1,sc2,sc3]
				writer.writerow(data)
		
		else:
			pass
			
		
	
data = ssc.socketTextStream("localhost", 6100)


j_data= data.map(outr).foreachRDD(func)


ssc.start()
ssc.awaitTermination()
