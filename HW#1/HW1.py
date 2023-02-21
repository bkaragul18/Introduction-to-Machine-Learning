#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd


# In[2]:


#load the whole dataset


# In[3]:


df = pd.read_csv("hw01_data_points.csv", header= None)


# In[4]:


cl = pd.read_csv("hw01_class_labels.csv", header= None)


# In[5]:


K = cl.iloc[:,0].max() #define the number of classes


# In[6]:


D = df.shape[1] #define the number of columns (in this case number of columns refers the nucletide length)


# In[7]:


C = ['A','C','G','T'] 


# In[8]:


train_data = df.iloc[0:300,:] #Divide the nucleotide dataset to train dataset


# In[9]:


test_data = df.iloc[300:,:] #Divide the nucleotide dataset to train dataset


# In[10]:


cl_train = cl.iloc[0:300] #Divide the labels dataset to train dataset


# In[11]:


cl_test = cl.iloc[300:] #Divide the labels dataset to test dataset


# Apply the parameter estimation functions

# In[12]:


pAcd = [[np.round((np.sum(np.multiply(train_data[train_data == 'A'].replace('A',1).fillna(0).iloc[:,i].values.tolist() 
                           ,(cl_train== (j+1)).astype(int).T.iloc[0,:].values.tolist())))
         /(sum((cl_train== (j+1)).astype(int).T.iloc[0,:].values.tolist())), decimals=8) for i in range(D)] for j in range(K)]


# In[13]:


print(pAcd)


# In[14]:


pCcd = [[np.round((np.sum(np.multiply(train_data[train_data == 'C'].replace('C',1).fillna(0).iloc[:,i].values.tolist() 
                           ,(cl_train== (j+1)).astype(int).T.iloc[0,:].values.tolist())))
         /(sum((cl_train== (j+1)).astype(int).T.iloc[0,:].values.tolist())), decimals=8) for i in range(D)] for j in range(K)]


# In[15]:


print(pCcd)


# In[16]:


pGcd = [[np.round((np.sum(np.multiply(train_data[train_data == 'G'].replace('G',1).fillna(0).iloc[:,i].values.tolist() 
                           ,(cl_train== (j+1)).astype(int).T.iloc[0,:].values.tolist())))
         /(sum((cl_train== (j+1)).astype(int).T.iloc[0,:].values.tolist())), decimals=8) for i in range(D)] for j in range(K)]


# In[17]:


print(pGcd)


# In[18]:


pTcd = [[np.round((np.sum(np.multiply(train_data[train_data == 'T'].replace('T',1).fillna(0).iloc[:,i].values.tolist() 
                           ,(cl_train== (j+1)).astype(int).T.iloc[0,:].values.tolist())))
         /(sum((cl_train== (j+1)).astype(int).T.iloc[0,:].values.tolist())), decimals=8) for i in range(D)] for j in range(K)]


# In[19]:


print(pTcd)


# In[20]:


class_priors = [np.mean(cl_train.iloc[:,0] == (i + 1)) for i in range(K)]


# In[21]:


print(class_priors)


# In[22]:


def score_func(dataset):
    score_result = []
    for c in range(K):
        individual_score = 1
        result = []
        for i in range(dataset.shape[0]):
            for j in range(D):
                if(dataset.iloc[i,j] == 'A'):
                    individual_score *= pAcd[c][j]
                elif (dataset.iloc[i,j] == 'G'):
                    individual_score *= pGcd[c][j]
                elif (dataset.iloc[i,j] == 'C'):
                    individual_score *= pCcd[c][j]
                elif (dataset.iloc[i,j] == 'T'):
                    individual_score *= pTcd[c][j]
            result.append(np.log(individual_score) + np.log(class_priors[c]))
            individual_score = 1
        score_result.append(result)
    return score_result


# In[23]:


def compare_scores(score_list):
    class_result = []
    for i in range(len(score_list[0])):
        if(score_list[0][i] > score_list[1][i]):
            class_result.append(1)
        else:
            class_result.append(2)
    return class_result


# In[24]:


train_scores = compare_scores(score_func(train_data))


# In[25]:


test_scores = compare_scores(score_func(test_data))


# In[26]:


def get_confusion_matrix(scores,labels):
    trth1prd1 = 0
    trth1prd2 = 0
    trth2prd1 = 0
    trth2prd2 = 0
    for i in range(labels.shape[0]):
        if(labels.iloc[i].values[0] == scores[i]):
            if(scores[i] == 1):
                trth1prd1 +=1
            else:
                trth2prd2 +=1
        else:
            if(scores[i] == 1):
                trth2prd1 +=1
            else:
                trth1prd2 +=1
    return "y_truth    1     2\ny_pred\n1          {}    {}\n2          {}     {}".format(trth1prd1,trth2prd1,trth1prd2,trth2prd2) 


# In[27]:


confusion_train = get_confusion_matrix(train_scores,cl_train)


# In[28]:


confusion_test = get_confusion_matrix(test_scores,cl_test)


# In[29]:


print(confusion_train)


# In[30]:


print(confusion_test)

