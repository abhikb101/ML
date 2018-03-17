import numpy as np
from sklearn import cross_validation,neighbors
import pandas as pd
df=pd.read_csv('data.txt')
df.replace('?',-99999, inplace=True)
df.drop('id',axis=1,inplace=True)
x=np.array(df.drop(['class'],1))
y=np.array(df['class'])
x_tr,x_ts,y_tr,y_ts=cross_validation.train_test_split(x,y,test_size=0.2)
clf=neighbors.KNeighborsClassifier()
clf.fit(x_tr,y_tr)
print(clf.score(x_ts,y_ts))
