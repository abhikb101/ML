#importing libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
#importing data
df=pd.read_csv('C:/Users/HP-PC/Desktop/oct/criminal predit/criminal_train.csv')
test=pd.read_csv('C:/Users/HP-PC/Desktop/oct/criminal predit/criminal_test.csv')
#checking for null values
print(df.isnull().sum())
y=df['Criminal']
print(df.describe())
x=df
#features with very small sd are removed beacuse they have very less effect on the ans 
new=['PERID','MAIIN102','AIIND102','IIHHSIZ2','IIKI17_2','IRHH65_2','IIHH65_2','VEREP','Criminal','PRXRETRY',]
new1=['PERID','MAIIN102','AIIND102','IIHHSIZ2','IIKI17_2','IRHH65_2','IIHH65_2','VEREP','PRXRETRY',]
x=x.drop(new,1)
#random forest has been applied to classify
clf=RandomForestClassifier()
clf.fit(x,y)
ans=clf.predict(test.drop(new1,1))
aa=test
aa['Criminal']=pd.DataFrame(ans,columns=['Criminal'])
anss=aa[['PERID','Criminal']]
anss.to_csv('sol.csv')
