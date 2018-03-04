#importing libraries
import pandas as pd
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
#loading  data
df = pd.read_csv('C:\\Users\HP-PC\PycharmProjects\\untitled3\wiki.csv', encoding='utf-8')
#removing unnecessary features
df=df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]
#adding new features
df['HL_PCT']=(df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.0
df['PCT']=(df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df=df[['Adj. Close','HL_PCT','PCT','Adj. Volume']]
forecast_col='Adj. Close'
df.fillna(-9999, inplace=True)
#creating the label
forecast_out=int(math.ceil(0.01*len(df)))
df['label']=df[forecast_col].shift(-forecast_out)
#droping data without label
df.dropna(inplace=True)
y=df['label']
x=df.drop(['label'],axis=1)
#x=preprocessing.scale(x)
#splitting data into training and testing data
x_train,x_test,y_train,y_test=cross_validation.train_test_split(x,y,test_size=0.2)
#fitting through linear regression
clf = LinearRegression(n_jobs=-1)
clf.fit(x_train,y_train)
#checking the accuracy
accuracy=clf.score(x_test,y_test)

