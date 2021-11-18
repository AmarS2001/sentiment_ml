import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from sklearn.preprocessing import StandardScalar
import time
######################################################

sc=SparkContext("local","ml_spark")
sc.setLogLevel('OFF')
ssc=StreamingContext(sc,1)
sp=SparkSession(sc)

def output(rdd):
  df=sp.read.json(rdd)
  print(df)
  #df.printSchema()
  #df.show()
  

lines=ssc.socketTextStream("locahost",6100)
lines.foreachRDD(lambda rdd:output(rdd))

ssc.start()
ssc.awaitTermination()
