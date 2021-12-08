import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if sys.argv[1] == 'test':
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

else:
	df2= pd.read_csv("cluster_track.csv")
	x=np.arange(len(df2))
	y1=np.array(df2['centroid1_diff'])
	y2=np.array(df2['centroid2_diff'])
	
	plt.plot(x,y1)
	plt.plot(x,y2)
	
	plt.xlabel('batches ->')  
	plt.ylabel('centroid_shift ->')

	plt.legend(['centroid_shift 1', 'centroid_shift 2'])

	plt.show()

