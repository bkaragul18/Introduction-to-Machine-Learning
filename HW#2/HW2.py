#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats


# In[2]:


data_set = np.genfromtxt("hw02_data_points.csv", delimiter= ',')


# In[3]:


class_labels = np.genfromtxt("hw02_class_labels.csv", delimiter= ',')


# In[4]:


N = data_set.shape[0]

D = data_set.shape[1]

K = np.max(class_labels).astype(int)

train_data = data_set[:10000,:]
test_data = data_set[10000:,:]
train_labels = class_labels[:10000].astype(int)
test_labels = class_labels[10000:].astype(int)

# one-of-K encoding
Y_truth = np.zeros((N, K)).astype(int)
Y_truth[range(N), class_labels[:].astype(int) -1] = 1

Y_truth_train = Y_truth[:10000,:]
Y_truth_test = Y_truth[10000:,:]


# In[5]:


def sigmoid(X, w, w0):
    return(1 / (1 + np.exp(-(np.matmul(X, w) + w0))))


# In[6]:


"""For predictions we use sigmoid function and gradients are calculated by derivaties of 
the sigmoid of error function with respect to W and W0.
- Derivative of error function with respect to W is (y_true-y_pred)*y_pred*(1-y_pred)*X
- Derivative of error function with respect to W0 is (y_true-y_pred)*y_pred*(1-y_pred)
"""

    
def gradient_W(X, y_truth, y_predicted):
    """ Shape of the created is 10x780, to use this matrix to update W parameter 
        we should transpose it so that having same dimensions with W matrix
    """
    return(np.asarray([-np.matmul((y_truth[:, c] - y_predicted[:, c])*y_predicted[:, c]*(1 - y_predicted[:, c]), X) for c in range(K)]).transpose())

def gradient_W0(y_truth, y_predicted):
    return(-np.sum((y_truth - y_predicted)*y_predicted*(1 - y_predicted), axis = 0))


# In[7]:


eta = 0.00001
iteration_count = 1000
W = np.genfromtxt("hw02_W_initial.csv", delimiter= ',')
W0 = np.genfromtxt("hw02_w0_initial.csv", delimiter= ',')


# In[8]:


iteration = 1
objective_values = []
while (iteration <= iteration_count):
    Y_predicted_train = sigmoid(train_data,W,W0)
    
    objective_values = np.append(objective_values, (0.5*np.sum((Y_truth_train - Y_predicted_train)**2)))
    
    W_old = W
    W0_old = W0
        
    W = W - eta*gradient_W(train_data,Y_truth_train, Y_predicted_train)
    W0 = W0 - eta*gradient_W0(Y_truth_train, Y_predicted_train)
    
    iteration += 1


# In[9]:


print(W)


# In[10]:


print(W0)


# In[11]:


# plot objective function during iterations
plt.figure(figsize = (7, 3))
plt.plot(range(1, iteration_count + 1), objective_values, "k-")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.show()


# In[12]:


y_predicted_train_arr = np.argmax(Y_predicted_train, axis = 1) + 1

confusion_train = pd.crosstab(y_predicted_train_arr, train_labels.T,
                               rownames = ["y_pred"],
                               colnames = ["y_truth"])
print(confusion_train)


# In[13]:


Y_predicted_test = sigmoid(test_data,W,W0)

y_predicted_test_arr = np.argmax(Y_predicted_test, axis = 1) + 1

confusion_test = pd.crosstab(y_predicted_test_arr, test_labels.T,
                               rownames = ["y_pred"],
                               colnames = ["y_truth"])
print(confusion_test)


# In[ ]:




