#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sklearn
from sklearn import svm
from sklearn import metrics
import pandas as pd


# In[2]:


# Read in the data
data_all = pd.read_csv('SVMSimpleFeatures_M.csv')


# In[3]:


# Split into features and target
x = data_all[['NumVoicedPerSec','PercentVProxy', 'MeanPitch', 'StdPitch', 'HNR', 'Jitter','Shimmer', 'NHR']]
# Set severity values 0 = normal, 1 = dysarthric
data_all.loc[data_all.Severity != 0, 'Severity'] = 1
y = data_all['Severity'].values.ravel()

# Print first 5 rows to be safe
print(x.head(5))
print(y[1:6])


# In[4]:


# Get dimensions (to be safe again)
print(x.shape)
# Each vector along x-axis -> number of samples
# Each feature of vector along y-axis -> number of features
print(y.shape)
# Y is one dimensional (num samples,1)


# In[5]:


# Let's do it in a loop
avg_acc = 0
avg_bacc = 0
avg_precision = 0
num_iter = 10
for i in range(num_iter):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.5)
    # Create the support vector classifier
    clf = svm.SVC(kernel = 'linear')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("============= Run " + str(i) + " =============")
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    print("Balanced accuracy:",metrics.balanced_accuracy_score(y_test, y_pred))
    print("Precision:",metrics.precision_score(y_test, y_pred))
    avg_acc += metrics.accuracy_score(y_test, y_pred) 
    avg_bacc += metrics.balanced_accuracy_score(y_test, y_pred) 
    avg_precision += metrics.precision_score(y_test, y_pred) 


# In[6]:


print('Average Accuracy : ' + str(avg_acc/num_iter))
print('Average Balanced Accuracy : ' + str(avg_bacc/num_iter))
print('Average Precision : ' + str(avg_precision/num_iter))