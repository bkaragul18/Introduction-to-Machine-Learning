#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("./hw03_data_set.csv")


# In[3]:


#Divide the datasets as train and test
df_train = df.iloc[:150,:]
df_test = df.iloc[150:,:]

#Divide x and y axis
x_train = df_train.iloc[:,0]
y_train = df_train.iloc[:,1]

x_test = df_test.iloc[:,0]
y_test = df_test.iloc[:,1]

#Define parameters
bin_width = 0.37
origin = 1.5


# In[4]:


point_colors = np.array(["blue","red"])
minimum_value = origin
maximum_value = 5
data_interval = np.linspace(start=minimum_value,
                            stop=maximum_value,
                            num= 1400)


# In[5]:


#Define other parameters
N_train = len(x_train)
N_test = len(x_test)

left_borders = np.arange(start=minimum_value,
                         stop= maximum_value,
                         step=bin_width)
right_borders = np.arange(start=minimum_value+bin_width,
                         stop= maximum_value+bin_width,
                         step=bin_width)


# In[6]:


g_x = np.asarray([np.sum(((left_borders[i]< x_train) & (x_train <= right_borders[i]))*y_train)
             for i in range (len(left_borders))]) / np.asarray([np.sum(((left_borders[i]< x_train) & (x_train <= right_borders[i]))) for i in range (len(left_borders))])


# In[7]:


plt.figure(figsize = (10, 6))
for i in range(len(left_borders)):
    plt.plot([left_borders[i], right_borders[i]], [g_x[i], g_x[i]], "k-")
for i in range(len(left_borders) - 1):
    plt.plot([right_borders[i], right_borders[i]], [g_x[i], g_x[i + 1]], "k-")
plt.plot(x_train,y_train,"b.",label = "training", )
plt.plot(x_test, y_test,"r.",label = "test")
plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time to next eruption (min)")
plt.legend()
plt.show()


# In[8]:


#find the frequency of the each bin
bin_frequency = np.asarray([(((left_borders[i]< x_test) & (x_test <= right_borders[i]))) for i in range (len(left_borders))])


# In[9]:


#create a estimation array
def expand_array(array,frequency):
    result = np.zeros(N_test)
    for i in range(len(bin_frequency)):
        result += np.multiply(bin_frequency[i].astype(int),np.repeat(array[i],len(bin_frequency[i])))
    return result


# In[10]:


#RMSE
rmse = np.sqrt(sum(np.square(y_test - expand_array(g_x,bin_frequency)))/N_test)


# In[11]:


print("Regressogram => RMSE is "+str(rmse)+" when h is "+ str(bin_width))


# --------------------------------------------------------------------------------

# In[12]:


g_x = np.asarray([np.sum(((left_borders[i]< x_train) & (x_train <= right_borders[i]))*y_train)
                  for i in range (len(left_borders))]) / np.asarray([np.sum(((left_borders[i]< x_train) & (x_train <= right_borders[i]))) for i in range (len(left_borders))])


# In[13]:


g_x = np.asarray([((np.abs((x-x_train)/bin_width) <= 1/2).astype(int)*y_train).sum() for x in data_interval])/np.asarray([((np.abs((x-x_train)/bin_width) <= 1/2).astype(int)).sum() for x in data_interval])


# In[14]:


plt.figure(figsize = (10, 6))
plt.plot(data_interval,g_x, "k-")
plt.plot(x_train,y_train,"b.",label = "training", )
plt.plot(x_test, y_test,"r.",label = "test")
plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time to next eruption (min)")
plt.legend()
plt.show()


# In[15]:


g_x_test = np.asarray([((np.abs((x-x_train)/bin_width) <= 1/2).astype(int)*y_train).sum() for x in x_test])/np.asarray([((np.abs((x-x_train)/bin_width) <= 1/2).astype(int)).sum() for x in x_test])


# In[16]:


#RMSE
rmse = np.sqrt(sum(np.square(y_test-g_x_test)/N_test))


# In[17]:


print("Running Mean Smoother => RMSE is "+str(rmse)+" when h is "+ str(bin_width))


# --------------------------------------------------------------------------------

# In[18]:


g_x = np.asarray([(((1/np.sqrt(2*np.pi))*np.exp(-np.square(((x-x_train)/bin_width))/2))*y_train).sum() for x in data_interval])/np.asarray([((1/np.sqrt(2*np.pi))*np.exp(-np.square(((x-x_train)/bin_width))/2)).sum() for x in data_interval])


# In[19]:


plt.figure(figsize = (10, 6))
plt.plot(data_interval,g_x, "k-")
plt.plot(x_train,y_train,"b.",label = "training", )
plt.plot(x_test, y_test,"r.",label = "test")
plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time to next eruption (min)")
plt.legend()
plt.show()


# In[20]:


g_x_test = np.asarray([(((1/np.sqrt(2*np.pi))*np.exp(-np.square(((x-x_train)/bin_width))/2))*y_train).sum() for x in x_test])/np.asarray([((1/np.sqrt(2*np.pi))*np.exp(-np.square(((x-x_train)/bin_width))/2)).sum() for x in x_test])


# In[21]:


#RMSE
rmse = np.sqrt(sum(np.square(y_test-g_x_test)/N_test))


# In[22]:


print("Kernel Smoother => RMSE is "+str(rmse)+" when h is "+ str(bin_width))

