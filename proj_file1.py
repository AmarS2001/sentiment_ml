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
  return json.loads(rdd)

def func(rdd):
  print(rdd.collect())
  
  

lines=ssc.socketTextStream("locahost",6100)
j_lines= lines.map(output).foreachRDD(func)

ssc.start()
ssc.awaitTermination()
