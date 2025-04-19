#!/usr/bin/env python
# coding: utf-8

# # Wine Quality Analysis

# ## Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm


# ## Data Preparation

# In[2]:


np.random.seed(42)
n_samples = 1000


# In[3]:


data = {
    'alcohol' : np.random.normal(10,2,n_samples),
    'acidity' : np.random.normal(5,1,n_samples),
    'ph' : np.random.normal(3.2,0.3,n_samples),
    'quality' : np.random.randint(3,10,n_samples)
}


# In[4]:


data = pd.DataFrame(data)


# ## Feature Analysis

# In[6]:


selected_features = ['alcohol','acidity','ph']
stats = data[selected_features].describe().T[['mean','std']]
stats['variance'] = stats['std'] ** 2


# In[7]:


print("Descriptive Statistics\n")
stats


# ## Modeling Gaussian Distribution

# In[14]:


plt.figure(figsize=(12,4))
for i,feature in enumerate(selected_features):
    plt.subplot(1,3,i+1)
    sns.histplot(data[feature],bins=30,kde=True,stat='density',color='blue',alpha=0.6)
    x = np.linspace(data[feature].min(),data[feature].max(),100)
    plt.plot(x,norm.pdf(x,stats.loc[feature,'mean'],stats.loc[feature,'std']),'r')
    plt.title(f"Gaussian Fit for {feature}")

plt.tight_layout()
plt.show()


# ## Correlation Analysis

# In[15]:


plt.figure(figsize=(8,6))
sns.heatmap(data.corr(),annot=True,cmap='coolwarm',fmt='.2f')
plt.title("Feature Correlation heatmap")


# ## Evaluation

# In[16]:


correlation_with_quality = data.corr()['quality'].drop('quality')
print('Feature correlation with wine quality')
correlation_with_quality


# In[ ]:




