#!/usr/bin/env python
# coding: utf-8

# # Lars Lefgren; Matthew Lindquist and David Sims, (2012), Rich Dad, Smart Dad: Decomposing the Intergenerational Transmission of Income, Journal of Political Economy, 120, (2), 268 - 303

# Using data from the Lefgren et al study, I demonstrate simple OLS in python using the statsmodels library

# In[1]:


#importing modules....
from __future__ import division
import pandas as pd
import numpy as np
import statsmodels.api as sm


# In[2]:


#read in the data
hw3=pd.read_stata(r"C:\Users\Rodgers\Desktop\PhD courses\PhD courses\EconS 593 Cowan\nls80.dta")
df=pd.DataFrame(data=hw3)
df.head()


# In[3]:


#This code fills in the missing values in the fathers education columnn with zero's. If you don't do this python will not run this 
#code.
df["feduc"] = df["feduc"].fillna(0)


# In[4]:


#purge the rows that have inf's and NaNs
#df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
#Ignore this piece of code, I'm just toying around with the model to see how it perfoms


# In[5]:


#lets run an ols model with this data
#I run this OLS regression with missing values set to zero. I thought that this was better than purging all the row's with missing data
#from the set#
#The model here is a simple one:lnwage=a+b.feduc+e
a=df.lwage
b=df.feduc
model=sm.OLS(a,b).fit()
model_prediciton=model.predict(b)
model_details=model.summary()
print(model_details)


# In[12]:


#IV regression.
#from statsmodels.sandbox.regression.gmm import IV2SLS
#import seaborn as sns
#lets estimate the following IV regression.
# Lets do this one lnwage=a +iq + educ + age + feduc+e
#We Will use this same equation to to the IV estimate ( Where we will asuume that feduc instruments for educ)


# In[6]:


#Multiple regression
# Lets do this one lnwage=a +iq + educ + age + feduc+e
from sklearn import linear_model
y=df['lwage']
x=df[['iq','age','tenure','feduc','educ']]
# with sklearn
regr = linear_model.LinearRegression()
regr.fit(x, y)
print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)


# In[7]:


#With statsmodels.api
x = sm.add_constant(x)
model=sm.OLS(y,x).fit()
model_prediciton=model.predict(x)
model_details=model.summary()
print(model_details)


# In[11]:


#Running OLS with the covariates we need to identify pi-1 and pi-2
import statsmodels.formula.api as smf
results = smf.ols('lwage ~ iq +age+tenure+feduc+educ',data=df).fit()
results_robust = results.get_robustcov_results()
print(results_robust.summary())


# In[ ]:


#There it is, simple OLS in python.

