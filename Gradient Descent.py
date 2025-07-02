#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("insurance_data.csv")
df.head()


# In[3]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df[['age','affordibility']],df.bought_insurance,test_size=0.2,random_state=25)



# In[4]:


len(X_train)


# In[5]:


X_train_scaled = X_train.copy()
X_train_scaled['age'] = X_train_scaled['age']/100

X_test_scaled = X_test.copy()
X_test_scaled['age'] = X_test_scaled['age'] /100


# In[6]:


X_train_scaled


# In[7]:


from keras.models import Sequential
from keras.layers import Dense


model = Sequential([
    Dense(1, input_shape=(2,), activation='sigmoid', 
          kernel_initializer='ones', bias_initializer='zeros')
])

model.compile(optimizer='adam',
             loss = 'binary_crossentropy',
             metrics=['accuracy'])

model.fit(X_train_scaled, y_train,epochs=5000)


# In[8]:


model.evaluate(X_test_scaled, y_test)


# In[9]:


X_test_scaled


# In[10]:


model.predict(X_test_scaled)


# In[11]:


y_test


# In[12]:


coef, intercept = model.get_weights()
coef, intercept


# In[13]:


def sigmoid(x):
    import math
    return 1/ (1+math.exp(-x))
sigmoid(18)


# In[14]:


def prediction_function(age, affordibility):
    weighted_sum = coef[0]*age + coef[1]* affordibility + intercept
    return sigmoid(weighted_sum)


# In[15]:


prediction_function(.47,1)


# In[16]:


prediction_function(18,1)


# In[17]:


def log_loss(y_true, y_predicted):
    epsilon = 1e-15
    y_predicted_new = [max(i,epsilon) for i in y_predicted]
    y_predicted_new = [min(i, 1-epsilon) for i in y_predicted_new]
    y_predicted_new = np.array(y_predicted_new)
    return -np.mean(y_true*np.log(y_predicted_new)+(1-y_true)*np.log(1-y_predicted_new))


# In[18]:


def sigmoid_numpy(X):
    return 1/(1+np.exp(-X))

sigmoid_numpy(np.array([12,0,1]))
                       


# In[35]:


class myNN:
    def __init__(self):
        self.w1 = 1
        self.w2 = 1
        self.bias = 0
    
    def fit(self, X ,y,epochs,loss_thresold):
        self.w1, self.w2,self.bias=self.gradient_descent(X['age'], X['affordibility'], y,epochs,loss_thresold)
        
    def predict(self, X_test):
        weighted_sum = self.w1
    def gradient_descent(self, age, affordibility, y_true,epochs):
    # w1w1 = w2 = 1
    bias = 0
    rate =0.5
    n = len(age)
    
    for i in range(epochs):
        weighted_sum = w1 * age + w2 * affordibility + bias
        y_predicted = sigmoid_numpy(weighted_sum)
        
        loss = log_loss(y_true, y_predicted)
        
        w1d = (1/n)*np.dot(np.transpose(age),(y_predicted-y_true))
        w2d = (1/n)*np.dot(np.transpose(affordibility),(y_predicted-y_true))
        
        bias_d = np.mean(y_predicted-y_true)
        
        w1 = w1 - rate * w1d
        w2 = w2 - rate * w2d
        bias = bias - rate * bias_d
        
        if i%50==0:
            print(f'Epoch:{i}, w1:{w2}, bias:{bias}, loss{loss}')
        if loss<=loss_thresold:
            break
    return w1, w2, bias
        


# In[32]:


customModel = myNN()
customModel.fit(X_train_scaled, y_train, epochs=500, loss_thresold=0.4631)


# In[ ]:




