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

#
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Flatten, Dense


from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
#from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from scikeras.wrappers import KerasRegressor
from sklearn.ensemble import AdaBoostRegressor

import pickle


# # Load prepared data

# In[42]:


# dir_path = "/Users/chikakoolsen/opt/python/thesis/code/tdcs_thesis/"
# dir_path = "/Users/mriworkshop/Documents/TDCS/code/tdcs_thesis/"
dir_path = "/Users/nei/duke/chikako/code/"
data_path = dir_path+"data/raw/"
img_path =  dir_path+"data/processed/"
model_path = dir_path+"models/"


# ## fmap mean all experiments

# In[3]:


file_mean = data_path+"fmap_mean_32to38.txt"
columns_mean =['exp', 'mini_exp', 'i', 'j', 'k', 'mean0', 'mean1', 'mean2', 'mean3', 'mean4', 'theory']
data = np.loadtxt(file_mean);


# In[4]:


df = pd.DataFrame(data, columns=columns_mean)
df = df.astype({"exp": int, "i": int, "j": int, "k": int, "mini_exp": int})



# # Split data

# ## Data 1. fmap mean all experiments

# In[6]:


# df1_train = df[~((df['exp']==36) & ((df['mini_exp']==5) | (df['mini_exp']==6)))]
# df1_val =  df[(df['exp']==36) & (df['mini_exp']==5)]
# df1_test =  df[(df['exp']==36) & (df['mini_exp']==6)]



# In[8]:


# X1_train = df1_train.iloc[:, 5:-1].values 
# y1_train = df1_train['theory'].values

# X1_test = df1_val.iloc[:, 5:-1].values 
# y1_test = df1_val['theory'].values

# X1_pred = df1_test.iloc[:, 5:-1].values 
# y1_pred = df1_val['theory'].values



# ## Data2: One experiment

# In[10]:

df_train = df[(df['exp']==36) & (df['mini_exp']!=6)]
df_test = df[(df['exp']==36) & (df['mini_exp']==6)]



# ## Data4: None zero

# In[28]:


df_nonzero = df[(df['mean0']!=0.0) & (df['mean1']!=0.0) & (df['mean2']!=0.0) & (df['mean3']!=0.0) & (df['mean4']!=0.0)]


# In[29]:


df_nonzero


# In[32]:


df4_train = df_nonzero[~((df_nonzero['exp']==36) & ((df_nonzero['mini_exp']==6) | (df_nonzero['mini_exp']==5)))]
df4_test =  df_nonzero[(df_nonzero['exp']==36) & (df_nonzero['mini_exp']==5)]
df4_pred =  df_nonzero[(df_nonzero['exp']==36) & (df_nonzero['mini_exp']==6)]


# In[33]:


X4_train = df4_train.iloc[:, 5:-1].values
y4_train = df4_train['theory'].values

X4_test = df4_test.iloc[:, 5:-1].values
y4_test = df4_test['theory'].values

X4_pred = df4_pred.iloc[:, 5:-1].values
y4_pred = df4_pred['theory'].values




# # Nonzero 32 to 38

# In[35]:


X_train = X4_train
y_train = y4_train
X_test = X4_test
y_test = y4_test


# In[63]:


shape = (len(X_train[0]),)
model = keras.Sequential()
model.add(Dense(10, activation='relu', input_shape=shape)) 
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adamax', loss='mse', metrics=["accuracy"])


# In[64]:


history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=[X_test, y_test])


# In[65]:


train_pred = model.predict(X_train)
print(mse(train_pred, y_train))
print(mape(train_pred, y_train))
test_pred = model.predict(X_test)
print(mse(test_pred, y_test))
print(mape(test_pred, y_test))


# In[66]:


model.evaluate(X_test, y_test)


# In[67]:


np.corrcoef(test_pred.flatten(), y_test)


# In[68]:


model.summary()


# In[69]:


history.history


# In[73]:


file = model_path+'model_nonzero_32to38.sav'
pickle.dump(model, open(file, 'wb'))



# ## Plot

# ### Loss function (MSE)

# In[70]:


model_df = pd.DataFrame(history.history)
model_df[['loss', 'val_loss']].plot()
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Over Training Period", pad=12);
plt.savefig(img_path+'loss_32to38.png')

# ### Accuracy

# In[71]:


model_df[['accuracy', 'val_accuracy']].plot()
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuray Over Training Period", pad=12);
plt.savefig(img_path+'accuracy_32to38.png')

# ### Theory vs Predict

# In[72]:


x = y_test.flatten()
y = test_pred.flatten()
m, b, r, p, st_er = stats.linregress(x,y) 

yfit = [b + m * xi for xi in x]
yisx = [0 + 1 * xi for xi in x]
plt.plot(x, yfit)
plt.plot(x, yisx)

plt.scatter(y_test, test_pred,  color='black')
plt.axis([0,100, 0, 100])
plt.xlabel("Theory (nT)")
plt.ylabel("Prediction (nT)")
plt.title("Neural Network Prediction vs Theory", fontsize=15)
# print(r, st_er)
print("r: {:.5f}, st_er: {:.6f}".format(r, st_er))
print("y = "+str(round(m,4))+"*x + "+str(round(b,4)))
plt.savefig(img_path+'theovspred_32to38.png')

# # Output data

# In[74]:


test_pred = model.predict(X4_pred)


# In[75]:


df4_pred['predict'] = test_pred




# In[78]:


df_test['predict'] = 0.00


# In[79]:


df_out = df_test[['i', 'j', 'k', 'predict']]
df_pre = df4_pred[['i', 'j', 'k', 'predict']]



# In[81]:


for x in range(len(df_pre)):
    i = df_pre.iloc[x, :]['i'].astype(int)
    j = df_pre.iloc[x, :]['j'].astype(int)
    k = df_pre.iloc[x, :]['k'].astype(int)
    pred = df_pre.iloc[x, :]['predict']
    idx = df_out[(df_out['i']==i) & (df_out['j']==j) & (df_out['k']==k)].index
    df_out.loc[idx, 'predict']= pred



# In[84]:


np.savetxt(img_path+"nn_nonzero_32to38.txt", df_out[['i', 'j', 'k', 'predict']], fmt="%i %i %i %s")



