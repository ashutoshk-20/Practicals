#!/usr/bin/env python
# coding: utf-8

# # Feed Forward Neural Network 

# ## Importing Libraries

# In[1]:


import tensorflow as tf
from tensorflow.keras.models import Sequential #type:ignore
from tensorflow.keras.layers import Dense #type:ignore
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ## Data Preparation

# ### Load and Spliting the data

# In[2]:


X,y = make_classification(
    n_samples = 1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    random_state=42
)


# In[3]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# ### Data preprocessing

# In[4]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ## Model Making

# ### Loading the model

# In[5]:


model = Sequential([
    Dense(64,activation='relu',input_shape=(X_train.shape[1],)),
    Dense(32,activation='relu'),
    Dense(1,activation='sigmoid'),
])


# ### Compiling the model

# In[7]:


model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# ### Training the model

# In[8]:


history = model.fit(X_train, y_train, 
                    epochs=20, 
                    batch_size=32, 
                    validation_split=0.2, 
                    verbose=1)


# ### Validation and evaluation

# In[10]:


test_loss,test_accuracy = model.evaluate(X_test,y_test,verbose = 0)
print(f"Test Accuracy : {test_accuracy*100:.2f}%")


# In[11]:


predictions = (model.predict(X_test)>0.5).astype('int32')
accuracy = accuracy_score(y_test,predictions)
print(f"Accuracy score : {accuracy*100:.2f}%")


# ## Visualization

# In[12]:


import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))
plt.plot(history.history['accuracy'], label="Train Accuracy")
plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[ ]:




