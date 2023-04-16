#!/usr/bin/env python
# coding: utf-8

# # GRIP: The Spark Foundation.
# # Data Science and Buisness Analytic Intern.
# # Task 1: Prediction Using Supervised ML.
# # Author: Dhanashree Suresh Koli.

# ##### In this task we have to predict the percentage score of a student baased on the no. of hours studied. The task has two variables where ythe feature is the no. of hours studied and the target value is the percentagr score. this can be solved using simple linear regression.

# # Step-1 import required libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data=pd.read_excel("C:\\Users\\Dhanashree\\OneDrive\\Desktop\\Data I.xlsx")
data


# # Step -2 Exploring the data

# In[3]:


data.shape


# In[4]:


data.describe()


# In[5]:


data.info()


# # Step-3 Data Visualiztion 

# In[6]:


data.plot(kind='scatter',x='hours',y='scores');
plt.show()


# In[7]:


data.corr(method='pearson')


# In[8]:


data.corr(method='spearman')


# In[9]:


hours=data['hours']
scores=data['scores']


# In[10]:


sns.distplot(hours)


# In[11]:


sns.distplot(scores)


# # Step-4 Linear Regression

# In[12]:


x=data.iloc[:,:-1].values
y=data.iloc[:,1].values


# In[13]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=50)
from sklearn.linear_model import LinearRegression
reg= LinearRegression()
reg.fit(x_train, y_train)


# In[14]:


m=reg.coef_
c=reg.intercept_
line=m*x+c
plt.scatter(x,y)
plt.plot(x,line);
plt.show()


# In[15]:


y_pred=reg.predict(x_test)
actual_predicted=pd.DataFrame({'Target':y_test,'predicted':y_pred})
actual_predicted


# In[16]:


sns.set_style('whitegrid')
sns.distplot(np.array(y_test-y_pred))
plt.show()


# In[17]:


h=9.25
s=reg.predict([[h]])
print("If student studies for {} hours per day he/she will score {}% in exam.".format(h,s))


# # Step-5 Model Evolution

# In[19]:


from sklearn import metrics
from sklearn.metrics import r2_score
print('mean absolute error:',metrics.mean_absolute_error(y_test,y_pred))
print('R2 score:',r2_score(y_test,y_pred))


# # Thank You! 

# In[ ]:




