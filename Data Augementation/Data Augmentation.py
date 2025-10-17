#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install opencv-python')


# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


# In[3]:


dataset_url = "http://download.tensorflow.org/example_images/flower_photos.tgz"
data_dir=tf.keras.utils.get_file('flower_photos', origin=dataset_url, cache_dir='.',untar=True)



# In[4]:


data_dir


# In[5]:


import pathlib
data_dir = pathlib.Path(data_dir)
data_dir 


# In[6]:


import os
print(os.getcwd())  # Shows the current directory


# In[7]:


from pathlib import Path
data_dir = Path("flower_photos")  # Change the path if needed
print(list(data_dir.glob('*/*.JPG')))


# In[8]:


from pathlib import Path

data_dir = Path(r"C:\Users\Archita Shrivastava\Downloads\flower_photos")  # Add 'r' before the string
print(data_dir.exists())  # Should return True if the folder exists



# In[9]:


print(list(data_dir.glob('*/*.*')))  # List all image files


# In[10]:


print(data_dir)


# In[11]:


print(data_dir.exists())  # Should return True
print(data_dir.is_dir())  # Should return True


# In[12]:


print(list(data_dir.glob('*')))  # Lists all items in the folder


# In[13]:


print(list(data_dir.iterdir()))


# In[14]:


list(data_dir.glob("**/*.JPG"))  # Correct way (recursive search)


# In[15]:


roses = list(data_dir.glob("roses/**/*.*"))  # Get all image formats in subdirectories
print(roses[:5])


# In[16]:


for folder in data_dir.iterdir():
    if folder.is_dir():
        print(folder)



# In[17]:


roses = list(data_dir.glob("**/roses/*.*"))  # Look for 'roses' anywhere
print(roses[:5])


# In[18]:


PIL.Image.open(str(roses[1]))


# In[19]:


tulips = list(data_dir.glob("**/tulips/*.*"))  # Look for 'tulips' anywhere
print(tulips)


# In[20]:


from PIL import Image

tulips = list(data_dir.glob("**/tulips/*.*"))

if tulips:
    img = Image.open(str(tulips[0]))  # Open the first image
    img.show()  # Display the image
else:
    print("No images found in the 'tulips' folder.")


# In[21]:


tulips = list(data_dir.glob("**/tulips/*.*"))  # Works regardless of folder depth


# In[22]:


tulips = list(data_dir.glob("**/tulips/*.*"))
PIL.Image.open(str(tulips[0]))


# In[23]:


flowers_images_dict = {
    'roses' : list(data_dir.glob("**/roses/*.*")),
    'daisy' : list(data_dir.glob("**/tulips/*.*")),
    'dandelion' : list(data_dir.glob("**/dandelion/*.*")),
    'sunflower' : list(data_dir.glob("**/sunflower/*.*")),
    'tulips' : list(data_dir.glob("**/tulips/*.*"))
}


# In[24]:


flowers_labels_dict = {
    'roses' : 0,
    'daisy' : 1,
    'dandelion' : 2,
    'sunflower' : 3,
    'tulips' : 4,
}


# In[25]:


str(flowers_images_dict['roses'][0])


# In[26]:


img = cv2.imread(str(flowers_images_dict['roses'][0]))
img


# In[27]:


img.shape


# In[28]:


cv2.resize(img,(180,180)).shape


# In[29]:


X,y = [] ,[]

for flower_name, images in flowers_images_dict.items():
    for image in images:
        img = cv2.imread(str(image))
        resized_img = cv2.resize(img,(180,180))
        X.append(resized_img)
        y.append(flowers_labels_dict[flower_name])
    


# In[30]:


X = np.array(X)
y = np.array(y)


# In[31]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=0)


# In[32]:


len(X_train)


# In[33]:


len(X_test)


# In[34]:


X_train_scaled = X_train/255
X_test_scaled = X_test / 255


# In[35]:


num_classes = 5  # Replace 5 with the actual number of categories in your dataset

print(num_classes)


# In[36]:


import tensorflow as tf
from tensorflow.keras import layers, models



# In[37]:


model = Sequential([
    layers.Conv2D(16, (3,3), padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, (3,3), padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')  # Use softmax for multi-class classification
])


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(X_train, y_train, epochs=30)


# In[38]:


model.evaluate(X_test_scaled, y_test)


# In[39]:


predictions = model.predict(X_test_scaled)
predictions


# In[40]:


score = tf.nn.softmax(predictions[0])
score


# In[41]:


np.argmax(score)


# In[42]:


y_test[0]


# In[44]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

data_augmentation = keras.Sequential([
    layers.RandomZoom(0.3),
])


# In[45]:


plt.axis('off')
plt.imshow(X[0])


# In[47]:


data_augmentation(X)[0]


# In[49]:


plt.axis('off')
plt.imshow(data_augmentation(X)[0].numpy().astype("uint8"))


# In[51]:


num_classes = 5

model = Sequential([
    data_augmentation,
    layers.Conv2D(16,3, padding='same', activation = 'relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32,3, padding='same', activation = 'relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64,3, padding='same', activation = 'relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation = 'relu'),
    layers.Dense(num_classes)
    
])

model.compile(optimizer = 'adam',
             loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             metrics=['accuracy'])
model.fit(X_train_scaled, y_train, epochs=30)


# In[ ]:




