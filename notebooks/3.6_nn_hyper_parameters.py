#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Flatten, Dense

from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
#from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from scikeras.wrappers import KerasRegressor
from sklearn.ensemble import AdaBoostRegressor


# # Load prepared data

# In[2]:

# dir_path = "/Users/chikakoolsen/opt/python/thesis/code/"
# dir_path = "/Users/mriworkshop/Documents/TDCS/code/"
dir_path = "/Users/nei/duke/chikako/"
save_path = dir_path+"tdcs_thesis/data/raw/"
img_path =  dir_path+"tdcs_thesis/data/processed/"
model_path =  dir_path+"tdcs_thesis/models/"
out_path = dir_path+"scp/"
# fname = "_32to38"
fname = "_32to38_erode"


# ## fmap mean all experiments

# In[4]:

file_mean = save_path+"fmap_mean_32to38.txt"
# file_mean = save_path+"fmap_mean_erode.txt"
columns_mean =['exp', 'mini_exp', 'i', 'j', 'k', 'mean0', 'mean1', 'mean2', 'mean3', 'mean4', 'theory']
data = np.loadtxt(file_mean);


# In[5]:

df = pd.DataFrame(data, columns=columns_mean)
df = df.astype({"exp": int, "i": int, "j": int, "k": int, "mini_exp": int})



# # Split data

# ## Data 1. fmap mean all experiments

# In[10]:
df1_train = df[~((df['exp']==36) & ((df['mini_exp']==5) | (df['mini_exp']==6)))]
df1_val =  df[(df['exp']==36) & (df['mini_exp']==5)]
df1_test =  df[(df['exp']==36) & (df['mini_exp']==6)]



# In[11]:

X1_train = df1_train.iloc[:, 5:-1].values 
y1_train = df1_train['theory'].values

X1_test = df1_val.iloc[:, 5:-1].values 
y1_test = df1_val['theory'].values

X1_pred = df1_test.iloc[:, 5:-1].values 
y1_pred = df1_val['theory'].values



# ## Data2: One experiment

# In[57]:


df_train = df[(df['exp']==36) & (df['mini_exp']!=6)]
df_test = df[(df['exp']==36) & (df['mini_exp']==6)]


X2_train = df_train.iloc[:, 5:-1].values
y2_train = df_train['theory'].values

X2_test = df_test.iloc[:, 5:-1].values
y2_test = df_test['theory'].values


# ## Data4: None zero

# In[23]:

df_nonzero = df[(df['mean0']!=0.0) & (df['mean1']!=0.0) & (df['mean2']!=0.0) & (df['mean3']!=0.0) & (df['mean4']!=0.0)]

df4_train = df_nonzero[~((df_nonzero['exp']==36) & ((df_nonzero['mini_exp']==6) | (df_nonzero['mini_exp']==5)))]
df4_test =  df_nonzero[(df_nonzero['exp']==36) & (df_nonzero['mini_exp']==5)]
df4_pred =  df_nonzero[(df_nonzero['exp']==36) & (df_nonzero['mini_exp']==6)]

# In[24]:

X4_train = df4_train.iloc[:, 5:-1].values
y4_train = df4_train['theory'].values

X4_test = df4_test.iloc[:, 5:-1].values
y4_test = df4_test['theory'].values

X4_pred = df4_pred.iloc[:, 5:-1].values
y4_pred = df4_pred['theory'].values




# # Neural Network

# ## Find out best layer and units

# In[197]:


X_train = X1_train
y_train = y1_train
X_test = X1_test
y_test = y1_test



# In[63]:

myCallbacks = [tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=10, mode='max')]


# In[172]:

def create_nn_corr(bsize, opt, epoch, isCallback=False):
    model = keras.Sequential()
    model.add(Dense(8, activation='relu', input_shape=(len(X_train[0]),)))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation="linear"))
    model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])
    if isCallback:
        model.fit(X_train, y_train, batch_size=bsize, epochs=epoch, callbacks=myCallbacks)
    else:
        model.fit(X_train, y_train, batch_size=bsize, epochs=epoch)
        
    y_predict = model.predict(X_test)
    evaluate = model.evaluate(X_test, y_test)
    corr = np.corrcoef(y_predict.flatten(), y_test)
    m, b, r, p, st_er = stats.linregress(y_test.flatten(), y_predict.flatten()) 
    
    loss = evaluate[0]
    acc = evaluate[1]
    coef = corr[0][1]
    
    return acc, loss, coef, st_er


# In[203]:


acc_arr_batch = []
loss_arr_batch = []
coef_arr_batch = []
err_arr_batch = []
batch_size = [32, 64, 128, 256, 512, 1024, 2048, 10000]
for i in range(len(batch_size)):
    print("##### Batch:"+str(batch_size[i])+" #####")
    acc, loss, coef, err = create_nn_corr(batch_size[i], 'adam', 100, True)
    acc_arr_batch.append(acc)
    loss_arr_batch.append(loss)
    coef_arr_batch.append(coef)
    err_arr_batch.append(err)


# In[219]:

print(acc_arr_batch)
print(loss_arr_batch)
print(coef_arr_batch)
print(err_arr_batch)


# In[221]:

best_batch = batch_size.index(max(coef_arr_batch))

acc_arr_opt = []
loss_arr_opt = []
coef_arr_opt = []
err_arr_opt = []
opts = ['Adam', 'Adadelta', 'Adamax', 'RMSprop',  'SGD']
for i in range(len(opts)):
    print("##### Optimizer:"+str(opts[i])+" #####")
    acc, loss, coef, err = create_nn_corr(best_batch, opts[i], 100, True)
    acc_arr_opt.append(acc)
    loss_arr_opt.append(loss)
    coef_arr_opt.append(coef)
    err_arr_opt.append(err)


# In[224]:

print(acc_arr_opt)
print(loss_arr_opt)
print(coef_arr_opt)
print(err_arr_opt)


# In[286]:

best_opt = opts.index(max(coef_arr_opt))

acc_arr_epoch = []
loss_arr_epoch = []
coef_arr_epoch = []
err_arr_epoch = []
epochs = [50, 100, 200, 300, 500, 1000]
for i in range(len(epochs)):
    print("##### Epoches:"+str(epochs[i])+" #####")
    acc, loss, coef, err = create_nn_corr(best_batch, best_opt, epochs[i], False)
    acc_arr_epoch.append(acc)
    loss_arr_epoch.append(loss)
    coef_arr_epoch.append(coef)
    err_arr_epoch.append(err)


# In[287]:

print(acc_arr_epoch)
print(loss_arr_epoch)
print(coef_arr_epoch)
print(err_arr_epoch)
