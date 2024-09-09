#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Imports

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


# In[2]:


df1=pd.read_csv('Admission_Predict.csv')


# In[5]:


df = df1.drop(columns='Serial No.')


# In[6]:





# In[ ]:





# In[7]:


df.skew()


# In[8]:


sns.pairplot(df,x_vars=['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA',
       'Research' ],y_vars='Chance of Admit ',size=5,kind='scatter')


# In[9]:


plt.figure(figsize=(20, 16))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


# In[10]:


features= ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA',
       'Research']
X =df[features]
y = df['Chance of Admit ']


# In[11]:


X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=5,test_size=0.3)


# In[12]:


from math import sqrt


# In[13]:


#training
lm=LinearRegression()
lm.fit(X_train,y_train)
y_pred_train=lm.predict(X_train)
q= r2_score(y_train,y_pred_train)

#adj r2
n=X_train.shape[0]
k=X_train.shape[1]

adj_r2=1-((1-q)*(n-1)/(n-k-1))
rmse=sqrt(mean_squared_error(y_train,y_pred_train))
print('Value of r2 is',{q})
print('Value of adj r2 is',{adj_r2})
print('Value of rmse is',{rmse})


# In[ ]:





# In[14]:


#test
y_pred_test=lm.predict(X_test)
q= r2_score(y_test,y_pred_test)
#adj r2
n=X_test.shape[0]
k=X_test.shape[1]

adj_r2=1-((1-q)*(n-1)/(n-k-1))
rmse=sqrt(mean_squared_error(y_test,y_pred_test))
print('Value of r2 is',{q})
print('Value of adj r2 is',{adj_r2})
print('Value of rmse is',{rmse})


# In[15]:

import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Assuming df is your DataFrame containing data
# df = ...

# Select features and target variable
Inputs = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research']

# Fit a linear regression model
#lm = LinearRegression()
#X = df[Inputs]
#y = df['Target_Variable']  # Replace 'Target_Variable' with the name of your target variable column
#lm.fit(X, y)

# Function to update model prediction when slider value changes
def update_input():
    slider_values = {}
    columns = st.columns(2)  # Divide the layout into two columns
    for i, input_name in enumerate(Inputs):
        with columns[i % 2]:  # Alternate between the two columns
            min_value = float(df[input_name].min())  # Convert min_value to float
            max_value = float(df[input_name].max())  # Convert max_value to float
            default_value = df[input_name].mean()
            st.write(f"{input_name}:")
            slider_values[input_name] = st.slider("", min_value, max_value, default_value, key=input_name)
    new_data = pd.DataFrame([slider_values])
    return new_data

# Display sliders and output
st.write("Adjust the sliders:")
new_data = update_input()
st.write("Updated DataFrame:")
st.write(new_data)

# Make prediction based on input values
prediction = lm.predict(new_data)
st.write("Predicted value:")
st.write(prediction)







