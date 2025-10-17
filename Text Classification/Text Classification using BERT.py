#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text


# In[2]:


import pandas as pd

df = pd.read_csv("spam.csv")
df.head(5)


# In[3]:


df.groupby('Category').describe()


# In[4]:


df['Category'].value_counts()


# In[5]:


747/4825


# In[6]:


df_spam = df[df['Category'] == 'spam']
df_spam.shape


# In[7]:


df_ham = df[df['Category'] == 'ham']
df_ham.shape


# In[8]:


df_ham_downsampled = df_ham.sample(df_spam.shape[0])
df_ham_downsampled.shape


# In[9]:


df_balanced = pd.concat([df_spam, df_ham])
df_balanced.shape


# In[10]:


df_balanced['Category'].value_counts()


# In[11]:


df_balanced.sample(5)


# In[12]:


df_balanced['spam'] = df_balanced['Category'].apply(lambda x:1 if x =='spam' else 0)
df_balanced.sample(10)


# In[13]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_balanced['Message'], df_balanced['spam'], stratify = df_balanced['spam'])


# In[14]:


X_train.head(4)


# In[15]:


bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")


# In[16]:


def get_sentence_embeding(sentences):
    preprocessed_text = bert_preprocess(sentences)
    return bert_encoder(preprocessed_text)['pooled_output']
    
get_sentence_embeding([
    "500$ discount, hurry up",
    "Bhavin, are you up for a volleyball game tomorrow?"    
])


# In[17]:


e = get_sentence_embeding([
    "banana", 
    "grapes",
    "mango",
    "jeff bezos",
    "elon musk",
    "bill gates"
]
)


# In[18]:


e


# In[19]:


from sklearn.metrics.pairwise import cosine_similarity

cosine_similarity([e[0]],[e[1]])


# In[20]:


cosine_similarity([e[0]],[e[3]])


# In[21]:


cosine_similarity([e[3]],[e[4]])


# In[22]:


# Bert layers
text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
preprocessed_text = bert_preprocess(text_input)
outputs = bert_encoder(preprocessed_text)

# Neural network layers
l = tf.keras.layers.Dropout(0.1, name="dropout")(outputs['pooled_output'])
l = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(l)

# Use inputs and outputs to construct a final model
model = tf.keras.Model(inputs=[text_input], outputs = [l])


# In[23]:


len(X_train)


# In[24]:


model.summary()


# In[25]:


METRICS = [
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall')
]

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=METRICS)


# In[ ]:


model.fit(X_train, y_train, epochs=10)


# In[ ]:




