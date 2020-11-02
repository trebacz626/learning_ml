#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.multiclass import T
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal, uniform


# In[3]:


from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1, cache=True)


# In[4]:


X = mnist["data"]
Y = mnist["target"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.125)


# In[5]:


scaler = StandardScaler()
X_scaled_train = scaler.fit_transform(X_train)
X_scaled_test = scaler.transform(X_test)


# In[6]:


linear_clf = LinearSVC()
linear_clf.fit(X_scaled_train, Y_train)
print("fitted")
predicted = linear_clf.predict(X_scaled_train)
accuracy_score(Y_train, predicted)


# In[20]:


svm_clf = SVC(decision_function_shape="ovr")
svm_clf.fit(X_scaled_train[:10000],Y_train[:10000])
print("trained");
Y_pred = svm_clf.predict(X_scaled_train)
accuracy_score(Y_train, Y_pred)


# In[8]:


Y_pred_test = svm_clf.predict(X_scaled_test)
accuracy_score(Y_test, Y_pred_test)


# In[27]:


params_dict = {"gamma": reciprocal(0.001, 1), "C": uniform(0.1, 100)}
random_search = RandomizedSearchCV(svm_clf, params_dict, n_iter = 128, verbose = 2, cv=3, n_jobs = -1)
random_search.fit(X_scaled_train[:1000], Y_train[:1000])


# In[28]:


random_search.best_estimator_


# In[30]:


random_search.best_score_


# In[31]:


random_search.best_estimator_.fit(X_scaled_train, Y_train)


# In[32]:


Y_pred_train = random_search.best_estimator_.predict(X_scaled_train)
accuracy_score(Y_train, Y_pred_train)


# In[34]:


Y_pred_test = random_search.best_estimator_.predict(X_scaled_test)
accuracy_score(Y_test, Y_pred_test)


# In[ ]:




