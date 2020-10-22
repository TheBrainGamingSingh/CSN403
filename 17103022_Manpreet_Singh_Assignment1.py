#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

df = pd.read_csv('Assignment1.csv')

X = df[df.columns[:-1]]
Y = df['y']


# In[2]:


def MPneuron(x, t=None, *args):
    '''
    returns 1 if sum(x) is >= t
    x is a boolean vector and t is an int.
    '''
    x = np.array(x)
    val = np.sum(x)

    if t is not None:
        if val >= t:
            return 1
        else:
            return 0
    return np.nan


# In[3]:


accuracies = {}
for threshold in range(11):
    Y_pred = X.apply(MPneuron,axis=1,args=([threshold]))
    accuracy = np.sum(Y_pred == Y) / Y.shape[0]
    accuracies[threshold] = accuracy
    print('For threshold value {}, the accuracy is {}.'.format(threshold,accuracy))


# In[4]:


import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


x_vals = list(accuracies.keys())
y_vals = list(accuracies.values())


# In[6]:


plt.figure(figsize=(12,6))
plt.plot(x_vals,y_vals,'.--',markersize=15)
plt.ylim(0,1)
plt.xticks(x_vals)
plt.xlabel('Threshold values')
plt.ylabel('Accuracy')
for i,j in zip(x_vals, y_vals):
    plt.text(i-0.25, j+0.05, str(j))
plt.show()


# In[ ]:
