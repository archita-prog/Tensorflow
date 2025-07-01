#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
X_train.shape


# In[3]:


X_test.shape


# In[4]:


y_train[:5]


# In[5]:


y_train = y_train.reshape(-1, )
y_train[:5]


# In[6]:


classes = ["airplane", "automobile","bird","cat","deer","dog","frog","horse","ship","truck"]


# In[7]:


classes[9]


# In[8]:


def plot_sample(X,y,index):
    plt.figure(figsize = (15,2))
    plt.imshow(X[index])
    plt.xlabel(classes[y[index]])


# In[9]:


plot_sample(X_train , y_train, 0)


# In[10]:


plot_sample(X_train, y_train,2)


# In[11]:


plot_sample(X_train, y_train,3)


# In[12]:


X_train = X_train/255
X_test = X_test/255


# In[13]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

ann = keras.Sequential([
    layers.Flatten(input_shape=(32, 32, 3)),
    layers.Dense(3000, activation='relu'),
    layers.Dense(1000, activation='relu'),
    layers.Dense(10, activation='softmax')  # Changed to softmax
])

ann.compile(optimizer='SGD',
            loss='sparse_categorical_crossentropy',  # Fixed typo
            metrics=['accuracy'])

ann.fit(X_train, y_train, epochs=5)



# In[14]:


ann.evaluate(X_test, y_test)


# In[15]:


from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
y_pred = ann.predict(X_test)
y_pred_classes = [np.argmax(element) for element in y_pred]

print("Classification Report: \n", classification_report(y_test, y_pred_classes))


# In[16]:


cnn = keras.Sequential([
    # cnn
    # dense
    layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # Changed to softmax
])


# In[17]:


cnn.compile(optimizer='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[18]:


cnn.fit(X_train, y_train, epochs=10)


# In[19]:


y_test = y_test.reshape(-1,)
y_test[:5]


# In[20]:


plot_sample(X_test, y_test,1)


# In[21]:


y_pred = cnn.predict(X_test)
y_pred[:5]


# In[22]:


np.argmax([5,12,167,2])


# In[23]:


y_classes = [np.argmax(element) for element in y_pred]
y_classes[:5]


# In[24]:


y_test[:5]


# In[25]:


plot_sample(X_test, y_test,1)


# In[26]:


classes


# In[29]:


plot_sample(X_test, y_test, 3)


# In[31]:


classes[y_classes[3]]


# In[32]:


print("Classification Report: \n", classification_report(y_test, y_classes))


# In[ ]:




