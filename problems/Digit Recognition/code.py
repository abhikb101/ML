
# coding: utf-8

# # Digital Image recognition using Knn

# ### Importing libraries

# In[3]:

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


# ### Importing Data

# In[2]:

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')


# ### using Knn as the classifier

# In[4]:

x=train.drop('label',1)
y=train['label']
clf= KNeighborsClassifier(n_jobs=-1)
clf.fit(x,y)


# ### Predicting the values with the model

# In[5]:

y=clf.predict(test)


# ### Exporting the results

# In[6]:

ans=pd.DataFrame(y,columns=['label'])
ans.to_csv('ans.csv')

