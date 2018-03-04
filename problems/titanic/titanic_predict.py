#data analysis
import pandas as pd
#visualization
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing,cross_validation
#importing datd
train=pd.read_csv('c:/Users/HP-PC/desktop/oct/python/train.csv')
test=pd.read_csv('c:/Users/HP-PC/desktop/oct/python/test.csv')
c=test.copy().filter(items='PassengerId')
#checking the data
print(train.head(5))
print(test.head(5))
#checking for null vaules
print('-'*20)
print(train.isnull().sum())
print('length training data',len(train))
print('%age of missing Age:',(177/len(train['Age']))*100)
print('%age of missing Cabin:',(687/len(train['Age']))*100)
print('%age of missing Embarked:',(2/len(train['Age']))*100)
#since, cabin has 77% missing values we will drop that varibale
#Embarked has 0.22% missing values so we can set it with the value which is most frequent
#visualizing the graph for Embarked and Age 
ax=test["Age"].hist(bins=20)
ax.set(xlabel='Age',ylabel='count')
sns.countplot(x='Embarked',data=train)
plt.show()
print('-'*40)
print('Mean Age:',train['Age'].mean(skipna=True))
train.drop('PassengerId',axis=1,inplace=True)
print('Median Age:',train['Age'].median(skipna=True))
#Age data is skewed so median is better option than mean
#filling the missing values
train['Age'].fillna(28,inplace=True)
train['Embarked'].fillna('S',inplace=True)
test['Age'].fillna(28,inplace=True)
test['Embarked'].fillna('S',inplace=True)
#removing useless variables
train.drop('Cabin',axis=1,inplace=True)
train.drop('Name',axis=1,inplace=True)
train.drop('Ticket',axis=1,inplace=True)
test.drop('PassengerId',axis=1,inplace=True)
test.drop('Cabin',axis=1,inplace=True)
test.drop('Name',axis=1,inplace=True)
test.drop('Ticket',axis=1,inplace=True)
#creating categorical variable for pclass and sex
train2=pd.get_dummies(train,columns=["Pclass"])
train3=pd.get_dummies(train2,columns=["Sex"])
train3.drop('Sex_male',axis=1,inplace=True)
train4=pd.get_dummies(train3,columns=["Embarked"])
test2=pd.get_dummies(test,columns=["Pclass"])
test3=pd.get_dummies(test2,columns=["Sex"])
test3.drop('Sex_male',axis=1,inplace=True)
test4=pd.get_dummies(test3,columns=["Embarked"])
cols=["Age","SibSp","Parch","Pclass_1","Pclass_2","Embarked_C","Embarked_S","Sex_female"]
x=train4[cols]
#x=preprocessing.scale(x)
y=train4["Survived"]
x_train,x_test,y_train,y_test=cross_validation.train_test_split(x,y,test_size=0.2)
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train,y_train)
print(logreg.score(x_test,y_test))
x2=test4[cols]
y2=logreg.predict(x2)
y2=y2.reshape(418,1)
d=pd.DataFrame(data=y2[:,0],columns=["Survived"])


