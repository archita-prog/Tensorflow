#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

y_predicted = np.array([1,1,0,0,1])
y_true = np.array([0.30,0.7,1,0,0.5])


# In[7]:


def mae(y_true, y_predicted):
    total_error = 0
    for yt, yp in zip(y_true, y_predicted):  # for running for loops parallelly
        total_error += abs(yt - yp)
    print("Total error:", total_error)
    mae = total_error / len(y_true)
    print("MAE" , mae)
    return mae


# In[8]:


mae(y_true, y_predicted)


# In[9]:


np.mean(np.abs(y_predicted,y_true))


# In[12]:


np.log([0.0000000000001])


# In[13]:


epsilon = 1e-15


# In[15]:


y_predicted_new = [max(i,epsilon) for i in y_predicted]
y_predicted_new


# In[16]:


y_predicted_new = [min(i,1-epsilon) for i in y_predicted]
y_predicted_new


# In[17]:


y_predicted_new = np.array(y_predicted_new)
np.log(y_predicted_new)


# In[19]:


-np.mean(y_true*np.log(y_predicted_new) + (1-y_true)*np.log(1-y_predicted_new))


# In[26]:


def log_loss(y_true, y_predicted):
    y_predicted_new = [max(i,epsilon) for i in y_predicted]
    y_predicted_new = [min(i,1-epsilon) for i in y_predicted_new]
    y_predicted_new = np.array(y_predicted_new)
    return -np.mean(y_true*np.log(y_predicted_new) + (1-y_true)*np.log(1-y_predicted_new))
    
    


# In[27]:


log_loss(y_true, y_predicted)


# In[ ]:




