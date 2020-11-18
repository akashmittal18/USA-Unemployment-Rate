# -*- coding: utf-8 -*-
"""
Created on Tue May 12 17:10:45 2020

@author: Abhi kamboj
"""
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
##%matplotlib inline
import matplotlib.dates as mdates
plt.style.use('ggplot')
import math



# unemployment rate in the US from the year of 1948 to 2017
df = pd.read_excel('USA umemployment rate 1948-2017.xlsx')
df.head()



# change the name 'value' to 'unemployment rate'.
df = df.rename(columns={'Value':'Unemployment Rate'})
df.head()



# create a new table for each year and its average unemployment rate

# 1, easy way
date=df.set_index('Year').groupby('Year').mean()
date.head()

# 2, fun way
data = pd.DataFrame(df['Year'].unique(), columns=['Year'])
data.head()



# find the avg unemployment rate for every year. sum the rate of every 12 months and find its mean
sum=0
avg=[]
n=0
for x in range(len(data)):
    for y in range(n,len(df)):
        if(df['Year'][y] == data['Year'][x]):
            sum += df['Unemployment Rate'][y]
        else:
            avg.append(sum/12)
            n=y
            sum=0
            break
        if(y == 839): # y will never reach 840, so without this condition, the else condition above will not be activate
            avg.append((sum/12))


avg[0:5]

# combine the data
data['Unemployment Rate'] = pd.DataFrame(avg, columns=['Unemployment Rate'])

# round the rate to 2 decimal place
data['Unemployment Rate'] = data['Unemployment Rate'].round(2)

data.head()



# graph the data
fig,ax = plt.subplots(figsize=(15,5))
ax.plot(data['Year'], data['Unemployment Rate'])

# show the year with more detail
ax.locator_params(nbins=70, axis='x')

# italic the values in the x axis
fig.autofmt_xdate()

plt.title('US unemployment rate from 1948 to 2017')
plt.show()



# recessions in the 1980s and around 2009

# we need to log transform the ‘y’ variable to a try to convert non-stationary data to stationary.
# This also converts trends to more linear trends

data['Unemployment Rate'] = np.log(data['Unemployment Rate'])
data.head()


data_set = data['Unemployment Rate'].values


len(data_set)


# There are 70 years data, and I will use every 30 years to predict the unemploment rate in the 31th year.

training_set = data_set[:50]


# prepare the training data, use every 30 data to predict the 31th. 0-29 -> 30, and 1-30 -> 31, ...
X_train = []
y_train = []
for i in range(30, len(training_set)):
    X_train.append(training_set[i-30:i])
    y_train.append(training_set[i])


# prepare the test set. Here will use the last 30 values in the training set to predic the first value in the test set
test_set = data_set[20:] # last 50 values
X_test = []
y_test = data_set[50:]
for i in range(30, 50):
    X_test.append(training_set[i-30:i]) # the first value here is the 20th value in data_set


from sklearn.linear_model import LinearRegression

lrm = LinearRegression()

lrm.fit(X_train, y_train)

pred_lrm = lrm.predict(X_test)

# reverse the values from log
for i in range(20):
    y_test[i] = math.exp(y_test[i])
    pred_lrm[i] = math.exp(pred_lrm[i])


# last 20 years
L20y = data['Year'][50:]



fig,ax = plt.subplots(figsize=(15,5))
one, = ax.plot(L20y, y_test, color='red')
two, = ax.plot(L20y, pred_lrm, color='blue')
plt.legend([one,two],['Original','Predicted'])
ax.locator_params(nbins=20, axis='x')

# looks like the trend is shifted. Expecially the time during the recession
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=1)
knn.fit(X_train, y_train)
pred_knn = knn.predict(X_test)

for i in range(20):
    pred_knn[i] = math.exp(pred_knn[i])


fig,ax = plt.subplots(figsize=(15,5))
one, = ax.plot(L20y, y_test, color='red')
two, = ax.plot(L20y, pred_knn, color='blue')
plt.legend([one,two],['Original','Predicted'])
ax.locator_params(nbins=20, axis='x')


from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor()
tree.fit(X_train, y_train)
pred_tree = tree.predict(X_test)

for i in range(20):
    pred_tree[i] = math.exp(pred_tree[i])



fig,ax = plt.subplots(figsize=(15,5))
one, = ax.plot(L20y, y_test, color='red')
two, = ax.plot(L20y, pred_tree, color='blue')
plt.legend([one,two],['Original','Predicted'])
ax.locator_params(nbins=20, axis='x')


from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(n_jobs=100)
rfr.fit(X_train, y_train)
pred_rfr = rfr.predict(X_test)

for i in range(20):
    pred_rfr[i] = math.exp(pred_rfr[i])



fig,ax = plt.subplots(figsize=(15,5))
one, = ax.plot(L20y, y_test, color='red')
two, = ax.plot(L20y, pred_rfr, color='blue')
plt.legend([one,two],['Original','Predicted'])
ax.locator_params(nbins=20, axis='x')



from sklearn.svm import SVR
svr = SVR()
svr.fit(X_train, y_train)
pred_svr = svr.predict(X_test)

for i in range(20):
    pred_svr[i] = math.exp(pred_svr[i])



fig,ax = plt.subplots(figsize=(15,5))
one, = ax.plot(L20y, y_test, color='red')
two, = ax.plot(L20y, pred_svr, color='blue')
plt.legend([one,two],['Original','Predicted'])
ax.locator_params(nbins=20, axis='x')



fig,ax = plt.subplots(figsize=(15,5))
a, = ax.plot(L20y, y_test, color='red')
b, = ax.plot(L20y, pred_lrm, color='blue')
c, = ax.plot(L20y, pred_knn, color='yellow')
d, = ax.plot(L20y, pred_tree, color='green')
e, = ax.plot(L20y, pred_rfr, color='orange')
f, = ax.plot(L20y, pred_svr, color='black')
plt.legend([a,b, c,d,e,f],['Original','Linear Regression', 'KNN', 'Tree', 'Random Forest', 'SVM'])
