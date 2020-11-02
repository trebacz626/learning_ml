#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal, uniform
from sklearn.metrics import mean_squared_error


# In[9]:


from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
X = housing["data"]
Y = housing["target"]


# In[15]:


print(len(X))
print(max(Y))


# In[11]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[53]:


svr = SVR(kernel="rbf")
svr.fit(X_train_scaled, Y_train)
Y_train_pred = svr.predict(X_train_scaled)
np.sqrt(mean_squared_error(Y_train, Y_train_pred))


# In[54]:


svr


# In[60]:


params_dict = {"gamma": uniform(0.05, 0.1), "C": uniform(0, 5)}
random_search = RandomizedSearchCV(svr, params_dict, n_iter = 24, verbose = 2, cv=3, n_jobs = -1)
random_search.fit(X_train_scaled, Y_train)


# In[61]:


random_search.best_estimator_


# In[62]:


Y_train_pred = random_search.best_estimator_.predict(X_train_scaled)
np.sqrt(mean_squared_error(Y_train, Y_train_pred))


# In[63]:


Y_test_pred = random_search.best_estimator_.predict(X_test_scaled)
np.sqrt(mean_squared_error(Y_test, Y_test_pred))


# In[64]:


Y_test_pred = svr.predict(X_test_scaled)
np.sqrt(mean_squared_error(Y_test, Y_test_pred))


# ### It turns out that default settings were pretty good

# In[ ]:




