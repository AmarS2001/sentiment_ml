import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df=pd.read_csv("train_track.csv")
x=np.arange(len(df))
y1=np.array(df['SVM_class'])
y2=np.array(df['NB_Class'])
y3=np.array(df['RF_class'])

plt.plot(x,y1)
plt.plot(x,y2)
plt.plot(x,y3)

plt.xlabel('batches ->')  
plt.ylabel('Score ->')

plt.legend(['SVM', 'Naive Bias', 'Random Forest'])

plt.show()

