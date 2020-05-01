#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sklearn
from sklearn import svm
from sklearn import metrics
import pandas as pd
import numpy as np
from IPython.display import display, HTML 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation


# In[2]:


# Read in the data
data_all = pd.read_csv('SVMSimpleFeatures.csv')


# In[3]:


# Take out the features

normal_mask = data_all['Severity']==0
impaired_mask = data_all['Severity']!=0

# Drop the severity and sex column 
data_all.drop('Severity', axis = 1, inplace=True)
data_all.drop('Sex', axis = 1, inplace=True)
# Get the normal data
x_normal = data_all[normal_mask] 
# Split the normal data into training and testing
x_normal_train, x_normal_test = sklearn.model_selection.train_test_split(x_normal, test_size=0.2, random_state=42)
# Get the abnormal/dysarthric data
x_impaired = data_all[impaired_mask]

# Print first 5 rows to be safe
print(x_normal.head(5))


# In[4]:


# Design and run the model
model = Sequential()
model.add(Dense(25, input_dim=x_normal.shape[1], activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(x_normal.shape[1])) # Multiple output neurons
model.compile(loss='mean_squared_error', optimizer='adam')
#model.compile(loss='mean_absolute_error', optimizer='adam')
model.fit(x_normal_train,x_normal_train,verbose=1,epochs=100)


# In[5]:


# Trained. Let's test
pred = model.predict(x_normal_test)
score1 = np.sqrt(metrics.mean_squared_error(pred,x_normal_test))
pred = model.predict(x_normal)
score2 = np.sqrt(metrics.mean_squared_error(pred,x_normal))
pred = model.predict(x_impaired)
score3 = np.sqrt(metrics.mean_squared_error(pred,x_impaired))
print(f"Out of Sample Score (RMSE): {score1}")
print(f"Insample Normal Score (RMSE): {score2}")
print(f"Abnormal Score (RMSE): {score3}")