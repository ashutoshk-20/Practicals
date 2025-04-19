#!/usr/bin/env python
# coding: utf-8

# # Digit Recognition using Mnist Data

# ## Importing libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.decomposition import PCA


# ## Data Preparation

# ### Loading dataset

# In[2]:


mnist = fetch_openml('mnist_784',version=1)


# In[3]:


X = mnist.data / 255.0
y = mnist.target.astype(int)


# ### Preprocessing the images / Flattening

# In[4]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# ## Model

# ### Development

# In[6]:


model = GaussianNB()


# ### Training and testing

# In[7]:


model.fit(X_train,y_train)
y_pred = model.predict(X_test)


# ### Evaluation

# In[8]:


print("Clssification Report\n")
classification_report(y_test,y_pred)


# In[9]:


print("Confusion Matrix")
confusion_matrix(y_test,y_pred)


# ## Visualization

# In[17]:


fig,axes = plt.subplots(2,5,figsize=(10,5))
for i,ax in enumerate(axes.flat):
    img = X_test.iloc[i].values.reshape(28,28)
    ax.imshow(img,cmap='gray')
    ax.set_title(f'Pred: {y_pred[i]}, True: {y_test.iloc[i]}')
    ax.axis('off')
plt.show()


# ## PCA Visualization

# In[13]:


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_test)
plt.figure(figsize=(15,10))
sns.scatterplot(x=X_pca[:,0],y=X_pca[:,1],hue=y_pred,palette='tab10',alpha=0.5)
plt.title("Decision Boundaries using PCA")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.show()


# In[ ]:




