#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as spa
import scipy.stats as stat


# In[2]:


X = np.genfromtxt("hw05_data_set.csv",delimiter=",")
means = np.genfromtxt("hw05_initial_centroids.csv",delimiter=",")


# In[3]:


K = means.shape[0]
N = X.shape[0]


# In[4]:


def update_memberships(means, X):
    # calculate distances between centroids and data points
    D = spa.distance_matrix(means, X)
    # find the nearest centroid for each data point
    memberships = np.argmin(D, axis = 0)
    return memberships


# INITIALIZE EM ALGORITHM

# In[5]:


memberships = update_memberships(means,X)
prior_prob = np.array([len(X[memberships == k])/N for k in range(K)])
covar = np.array([np.mat((X[memberships == k] - means[k])).T * np.mat((X[memberships == k] - means[k])) / len(X[memberships == k]) for k in range(K)])


# In[6]:


print("Initial Covariance Matrices are given below")
print(covar)
print("Prior probabilities are given below")
print(prior_prob)


# E-Step

# In[7]:


def e_step(prior_prob,means,covar):
    h = np.zeros((N,K))
    for i in range(N):
        for j in range(K):
            h[i][j] = stat.multivariate_normal.pdf(X[i],means[j],covar[j])*prior_prob[j]             /np.sum([stat.multivariate_normal.pdf(X[i],means[k],covar[k])*prior_prob[k] for k in range(K)])
    return h


# M-Step

# In[8]:


def m_step(h):
    for i in range(K):
        prior_prob[i] = sum([h[j][i] for j in range(N)])/N
        means[i] = sum([h[j][i]*X[j] for j in range(N)])/sum([h[j][i] for j in range(N)])
        covar[i] = sum([h[j][i]*(np.asmatrix(X[j]-means[i]).T@np.asmatrix(X[j]-means[i])) for j in range(N)])/sum([h[j][i] for j in range(N)])
    return prior_prob,means,covar


# Iteration Phase

# In[9]:


for i in range(100):
    h = e_step(prior_prob,means,covar)
    prior_prob,means,covar = m_step(h)
print("means are given below")
print(means)


# Plotting

# In[10]:


def plot_current_state(means, memberships, X, init_means,init_covs):
    cluster_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928",
                               "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6", "#ffff99"])
    if memberships is None:
        plt.plot(X[:, 0], X[:, 1], ".", markersize = 10, color = "black")
    else:
        for c in range(K):
            plt.plot(X[memberships == c, 0], X[memberships == c, 1], ".", markersize = 10,
                     color = cluster_colors[c])


    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")

    x1_interval = np.linspace(-8, +8, 1601)
    x2_interval = np.linspace(-8, +8, 1601)
    x1_grid, x2_grid = np.meshgrid(x1_interval, x2_interval)
    X_grid = np.vstack((x1_grid.flatten(), x2_grid.flatten())).T
    for i in range(K):
      est_points = stat.multivariate_normal.pdf(X_grid, mean = means[i],
                                        cov = covar[i])
      est_points=est_points.reshape((len(x1_interval), len(x2_interval)))
      plt.contour(x1_grid, x2_grid, est_points, levels = [0.05],
                  colors = "k", linestyles = "dashed")
      init_points = stat.multivariate_normal.pdf(X_grid, mean = init_means[i],
                                        cov = init_covs[i])
      init_points=init_points.reshape((len(x1_interval), len(x2_interval)))
      plt.contour(x1_grid, x2_grid, init_points, levels = [0.05],
                  colors = cluster_colors[i])
    plt.figure(figsize=(10,10))
    plt.show()


# Setting Initial Covariances and Means

# In[11]:


init_means = np.zeros_like(means)
init_covs = np.zeros_like(covar)

init_means[0] = [0.0,5.5]
init_means[1] = [-5.5,0.0]
init_means[2] = [0.0,0.0]
init_means[3] = [5.5,0.0]
init_means[4] = [0.0,-5.5]

init_covs[0] = [[4.8,0.0],[0.0,0.4]]
init_covs[1] = [[0.4,0.0],[0.0,2.8]]
init_covs[2] = [[2.4,0.0],[0.0,2.4]]
init_covs[3] = [[0.4,0.0],[0.0,2.8]]
init_covs[4] = [[4.8,0.0],[0.0,0.4]]


# In[12]:


pred_memberships = update_memberships(means,X)
plot_current_state(means,pred_memberships,X,init_means,init_covs)


# In[ ]:




