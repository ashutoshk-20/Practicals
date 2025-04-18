#!/usr/bin/env python
# coding: utf-8

# # Letter Recognition

# ### Importing Libraries

# In[2]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd


# ## Data Preparation

# ### Data Loading

# In[3]:


data = pd.read_csv("../data/letter-recognition.csv",header= None)


# In[4]:


data[0].unique()


# In[5]:


data = data[data[0].str.len() == 1]
data[0] = data[0].apply(lambda x: ord(x)- ord('A'))


# In[6]:


X = data.iloc[:,1:].values
y = data.iloc[:,0].values


# In[7]:


y = to_categorical(y,num_classes=26)


# ### Data Splitting

# In[8]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# ### Data Preprocessing

# In[9]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ## Model Creation

# ### Loading the Model

# In[10]:


model = Sequential([
    Dense(128,activation='relu',input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64,activation='relu'),
    Dropout(0.3),
    Dense(26,activation='softmax')
])


# ### Model Compilation

# In[11]:


model.compile(
    optimizer='adam',
    loss = 'categorical_crossentropy',
    metrics= ['accuracy']
)


# ### Traing the model

# In[12]:


history = model.fit(X_train,y_train,
                    epochs = 30,
                    batch_size = 32,
                    validation_split = 0.2,
                    verbose = 1
                   )


# ### Model Evaluation and validation

# In[13]:


test_loss,test_accuracy = model.evaluate(X_test,y_test,verbose=0)
print(f"Test Accuracy : {test_accuracy*100:.2f}%")


# In[14]:


predictions = model.predict(X_test)
predictions = predictions.argmax(axis=1)
y_true = y_test.argmax(axis=1)
print("Classification Report:\n")
print(classification_report(y_true,predictions,target_names=[chr(i) for i in range(ord('A'),ord('Z')+1)]))


# ## Visualization

# In[15]:


import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'],label='Train Accuracy')
plt.plot(history.history['val_accuracy'],label='Validation Accuracy')
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'],label='Train Loss')
plt.plot(history.history['val_loss'],label='Validation Loss')
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[ ]:




