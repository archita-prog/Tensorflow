#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("sonar_dataset.csv", header = None)
df.sample(5)


# In[3]:


df.shape


# In[4]:


df.isna().sum()


# In[5]:


df.columns


# In[6]:


df[60].value_counts()


# In[7]:


X = df.drop(60, axis=1)
y = df[60]
y.head()


# In[8]:


y = pd.get_dummies(y, drop_first=True)
y = y.astype(int)  # Convert True/False to 1/0


# In[9]:


y.sample(5)


# In[10]:


y.value_counts()


# In[11]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=1)


# In[12]:


X_train.shape, X_test.shape


# In[13]:


import tensorflow as tf
from tensorflow import keras


# In[14]:


model = keras.Sequential([
    keras.layers.Dense(60, input_dim=60, activation='relu'),
    keras.layers.Dense(30, activation = 'relu'),
    keras.layers.Dense(15, activation ='relu'),
    keras.layers.Dense(1, activation = 'sigmoid')
    
    
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs =100, batch_size=8)


# In[15]:


model.evaluate(X_test, y_test)


# In[16]:


y_pred = model.predict(X_test).reshape(-1)
print(y_pred[:10])


y_pred = np.round(y_pred)
y_pred = np.round(y_pred)
print(y_pred[:10])


# In[17]:


y_test[:10]


# In[18]:


from sklearn.metrics import confusion_matrix, classification_report

print(classification_report(y_test, y_pred))


# In[19]:


model = keras.Sequential([
    keras.layers.Dense(60, input_dim=60, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(30, activation = 'relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(15, activation ='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation = 'sigmoid')
    
    
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs =100, batch_size=8)


# In[20]:


model.evaluate(X_test, y_test)


# In[21]:


y_pred = model.predict(X_test).reshape(-1)
y_pred = np.round(y_pred)
from sklearn.metrics import confusion_matrix, classification_report

print(classification_report(y_test, y_pred))


# In[22]:


count_class_0, count_class_1


# In[ ]:




