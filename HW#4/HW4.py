#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


data_set = np.genfromtxt("hw04_data_set.csv",delimiter=",",skip_header=1)


# In[3]:


train = data_set[:150,:]
test = data_set[150:,:]

x_train = train[:,0]
y_train = train[:,1]
x_test = test[:,0]
y_test = test[:,1]

N_train = len(x_train)
N_test = len(x_test)


# In[4]:


def train_regression_tree(P):
    node_indices = {}
    is_terminal = {}
    need_split = {}

    node_features = {}
    node_splits = {}
    node_frequencies = {}

    # put all training instances into the root node
    node_indices[1] = np.array(range(N_train))
    is_terminal[1] = False
    need_split[1] = True
    D= x_train.shape[0]

    while True:
        split_nodes = [key for key , value in need_split.items()
                       if value == True]
        if len(split_nodes) == 0:
            break

        for split_node in split_nodes:
            data_indices = node_indices[split_node]
            need_split[split_node] = False
            if len(data_indices) <= P:
                is_terminal[split_node] = True
                node_frequencies[split_node] = y_train[data_indices].mean()
            else:
                is_terminal[split_node] = False
                unique_values = np.sort(np.unique(x_train[data_indices]))
                split_positions = (unique_values[1:] + unique_values[:-1]) / 2
                split_scores = np.repeat(0.0, len(split_positions))

                for s in range(len(split_positions)):
                    left_indices = data_indices[x_train[data_indices] > split_positions[s]]
                    right_indices = data_indices[x_train[data_indices] <= split_positions[s]]
                    split_scores[s]= np.sqrt((np.sum(np.square(y_train[i] - y_train[left_indices].mean()) for i in left_indices)/N_train + np.sum(np.square(y_train[j] - y_train[right_indices].mean()) for j in right_indices)/N_train)/len(data_indices))

                split_d = np.argmin(split_scores)
                node_features[split_node] = split_d
                node_splits[split_node] = split_positions[split_d]

                left_indices = data_indices[x_train[data_indices] > node_splits[split_node]]
                node_indices[2 * split_node] = left_indices
                is_terminal[2 * split_node] = False
                need_split[2 * split_node] = True

                right_indices = data_indices[x_train[data_indices] <= node_splits[split_node]]
                node_indices[2 * split_node + 1] = right_indices
                is_terminal[2 * split_node + 1] = False
                need_split[2 * split_node + 1] = True

    return is_terminal,node_frequencies,node_splits


# In[5]:


def predict_point(is_terminal,node_frequencies,node_splits,point):
    index = 1
    while True:
        if is_terminal[index]:
            return node_frequencies[index]
        else:
            if point > node_splits[index]:
                index = 2 * index
            else:
                index = 2 * index + 1


# In[6]:


def predict_interval(is_terminal,node_frequencies,node_splits,data_interval):
    y_predicted = np.repeat(0,len(data_interval))
    for i in range(len(data_interval)):
        index = 1
        while True:
            if is_terminal[index] == True:
                y_predicted[i] = node_frequencies[index]
                break
            else:
                if data_interval[i] > node_splits[index]:
                    index = 2 * index
                else:
                    index = 2 * index + 1
    return y_predicted


# In[7]:


x_min = min(np.min(x_train), np.min(x_test))
x_max = max(np.max(x_train), np.max(x_test))
data_interval = np.linspace(x_min, x_max, num=1400)

is_terminal,node_frequencies,node_splits = train_regression_tree(25)

y_predicted = [predict_point(is_terminal,node_frequencies,node_splits,x) for x in data_interval]


# In[8]:


plt.figure(figsize = (10, 5))
plt.plot(x_train,y_train,"b.",label = "training" )
plt.plot(x_test, y_test,"r.",label = "test")
plt.plot(data_interval,y_predicted, "k")
plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time to next eruption (min)")
plt.legend()
plt.show()


# In[9]:


train_pred = [predict_point(is_terminal,node_frequencies,node_splits,x) for x in x_train]
test_pred = [predict_point(is_terminal,node_frequencies,node_splits,x) for x in x_test]

rmse_train = np.sqrt(np.mean(np.square(y_train-train_pred)))
rmse_test = np.sqrt(np.mean(np.square(y_test-test_pred)))


# In[10]:


print("RMSE on training set is ", rmse_train,"when P is 25")
print("RMSE on test set is ", rmse_test,"when P is 25")


# In[11]:


p_vals = np.arange(5,51,5)
rmse_train_prunning = []
rmse_test_prunning = []

for p in p_vals:
    is_terminal,node_frequencies,node_splits = train_regression_tree(p)

    train_pred = [predict_point(is_terminal,node_frequencies,node_splits,x) for x in x_train]
    test_pred = [predict_point(is_terminal,node_frequencies,node_splits,x) for x in x_test]

    rmse_train = np.sqrt(np.mean(np.square(y_train-train_pred)))
    rmse_test = np.sqrt(np.mean(np.square(y_test-test_pred)))

    rmse_train_prunning.append(rmse_train)
    rmse_test_prunning.append(rmse_test)


# In[12]:


plt.figure(figsize = (6, 6))
plt.plot(p_vals,rmse_train_prunning,"b.-",label = "training" )
plt.plot(p_vals, rmse_test_prunning,"r.-",label = "test")
plt.xlabel("Pre-prunning Size (P)")
plt.ylabel("RMSE")
plt.legend()
plt.show()

