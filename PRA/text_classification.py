#!/usr/bin/env python
# coding: utf-8

# # Text Classification using Fetch_20newgroups

# ## Importing Libraries

# In[1]:


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix,classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ## Data Loading

# In[2]:


twety_train = fetch_20newsgroups(subset='train',shuffle=True)
twety_test = fetch_20newsgroups(subset='test',shuffle=True)


# ## Defining Pipeline

# In[3]:


text_clf = Pipeline([
    ('vect',CountVectorizer()),
    ('tfidf',TfidfTransformer()),
    ('clf',MultinomialNB())
])


# ## Model

# ### Training

# In[4]:


text_clf.fit(twety_train.data,twety_train.target)


# ### Predictions and evaluation

# In[5]:


predicted = text_clf.predict(twety_test.data)


# In[6]:


accuracy = np.mean(predicted == twety_test.target)
print(f"Accuracy : {accuracy}")


# In[7]:


print("Classification Report:\n")
classification_report(twety_test.target,predicted,target_names=twety_test.target_names)


# ### Visualization

# In[8]:


cm = confusion_matrix(twety_test.target,predicted)


# In[9]:


plt.figure(figsize=(10,10))
sns.heatmap(cm,annot=True,fmt='d',xticklabels=twety_test.target_names,yticklabels=twety_test.target_names)
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()


# In[ ]:




