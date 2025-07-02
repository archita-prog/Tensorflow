#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np


# In[2]:


(X_train, y_train ) , (X_test, y_test) = keras.datasets.mnist.load_data()


# In[3]:


len(X_train)


# In[4]:


len(X_test)


# In[5]:


X_train[0].shape


# In[6]:


X_train[0]


# In[7]:


plt.matshow(X_train[1])


# In[8]:


plt.matshow(X_train[6])


# In[9]:


plt.matshow(X_train[25])


# In[10]:


y_train[:5]


# In[11]:


X_train.shape


# In[12]:


X_train = X_train / 255
X_test = X_test / 255


# In[13]:


X_train[0]


# In[14]:


X_train_flattened = X_train.reshape(len(X_train), 28*28)
X_test_flattened = X_test.reshape(len(X_test), 28*28)


# In[15]:


X_test_flattened.shape


# In[16]:


# converting 2-d array to 1-d array
X_train_flattened[0]


# In[17]:


# creating a simple neural network
# layering
model = keras.Sequential([
    keras.layers.Dense(10, input_shape =(784,), activation = 'sigmoid')   
    
])
model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)
model.fit(X_train_flattened, y_train, epochs = 5)

'''values are not sclaed'''


# In[18]:


model.predict(X_test_flattened)


# In[19]:


model.evaluate(X_test_flattened, y_test)


# In[20]:


plt.matshow(X_test[0])


# In[21]:


y_predicted = model.predict(X_test_flattened)
y_predicted[0]


# In[22]:


np.argmax(y_predicted[1])


# In[23]:


y_predicted_labels = [np.argmax(i) for i in y_predicted]
y_predicted_labels[:5]


# In[24]:


y_test[:5]


# In[25]:


cm = tf.math.confusion_matrix(labels=y_test, predictions = y_predicted_labels)
cm


# In[26]:


import seaborn as sns
plt.figure(figsize = (10,7))
sns.heatmap(cm, annot= True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
'''Numbers in diagonals are  not in errors'''


# In[27]:


# creating a simple neural network
# layering
model = keras.Sequential([
    keras.layers.Dense(100, input_shape =(784,), activation = 'relu'), 
    keras.layers.Dense(10, activation = 'sigmoid')  
    
    
])
model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)
model.fit(X_train_flattened, y_train, epochs = 5)


# In[28]:


model.evaluate(X_test_flattened, y_test)


# In[29]:


plt.figure(figsize = (10,7))
sns.heatmap(cm, annot= True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[33]:


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(100, activation = 'relu'), 
    keras.layers.Dense(10, activation = 'sigmoid')  
    
    
])
tb_callback = tf.keras.callbacks.TensorBoard(log_dir="logs/", histogram_freq=1)
model.compile(
    optimizer = 'SGD',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)
model.fit(X_train, y_train, epochs = 5, callbacks=[tb_callback])


# In[31]:


get_ipython().system('pip show tensorboard')


# In[32]:


import tensorboard
print(tensorboard.__version__)


# In[34]:


get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().run_line_magic('tensorboard', '--logdir logs/fit')


# In[ ]:




