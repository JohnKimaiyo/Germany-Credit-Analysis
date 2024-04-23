#!/usr/bin/env python
# coding: utf-8

# # German Credit Data Analysis(Python)
# 
# ## Problem
# 
# ## 1. Determine the optimum age to target for customers
# 
# ## 2.Determine the type of loan that attracts most clients
# 
# 

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


german_credit_df = pd.read_csv(r"C:\Users\jki\Downloads\german_credit_data.csv")
german_credit_df.head(5)


# In[5]:


print(german_credit_df.columns)


# In[6]:


print("Purpose : ",german_credit_df.Purpose.unique())
print("Sex : ",german_credit_df.Sex.unique())
print("Housing : ",german_credit_df.Housing.unique())
print("Saving accounts : ",german_credit_df['Saving accounts'].unique())
print("Checking account : ",german_credit_df['Checking account'].unique())


# In[7]:


german_credit_df['Saving accounts'] = german_credit_df['Saving accounts'].map({"little":0,"moderate":1,"quite rich":2 ,"rich":3 });
german_credit_df['Saving accounts'] = german_credit_df['Saving accounts'].fillna(german_credit_df['Saving accounts'].dropna().mean())

german_credit_df['Checking account'] = german_credit_df['Checking account'].map({"little":0,"moderate":1,"rich":2 });
german_credit_df['Checking account'] = german_credit_df['Checking account'].fillna(german_credit_df['Checking account'].dropna().mean())

german_credit_df['Sex'] = german_credit_df['Sex'].map({"male":0,"female":1}).astype(float);

german_credit_df['Housing'] = german_credit_df['Housing'].map({"own":0,"free":1,"rent":2}).astype(float);

german_credit_df['Purpose'] = german_credit_df['Purpose'].map({'radio/TV':0, 'education':1, 'furniture/equipment':2, 'car':3, 'business':4,
       'domestic appliances':5, 'repairs':6, 'vacation/others':7}).astype(float);

german_credit_df.head(10)


# In[9]:


plt.scatter(german_credit_df['Credit amount'],german_credit_df["Age"])
plt.figure()


# In[12]:


sns.pairplot(german_credit_df)


# In[14]:


plt.scatter(german_credit_df['Credit amount'],german_credit_df["Duration"])
plt.figure()


# In[15]:


plt.scatter(german_credit_df['Saving accounts'],german_credit_df["Duration"])
plt.figure()


# In[16]:


fig = german_credit_df["Purpose"].hist(bins=8)
fig.text(-1, 150, 'Frequency', ha='center')
fig.text(0, -30, 'Radio', ha='center')
fig.text(1, -50, 'education', ha='center')
fig.text(2, -30, 'furniture', ha='center')
fig.text(3, -50, 'car', ha='center')
fig.text(4, -30, 'business', ha='center')
fig.text(5, -50, 'appliances', ha='center')
fig.text(6, -30, 'repairs', ha='center')
fig.text(7, -50, 'vacation', ha='center')


# In[17]:


limitedCredit = german_credit_df[(german_credit_df["Credit amount"]<=5000)==True];
imitedCredit = german_credit_df[(german_credit_df["Credit amount"]>2000)==True];
fig = limitedCredit["Purpose"].hist(bins=8)
fig.text(-1, 150, 'Frequency', ha='center')
fig.text(0, -30, 'Radio', ha='center')
fig.text(1, -50, 'education', ha='center')
fig.text(2, -30, 'furniture', ha='center')
fig.text(3, -50, 'car', ha='center')
fig.text(4, -30, 'business', ha='center')
fig.text(5, -50, 'appliances', ha='center')
fig.text(6, -30, 'repairs', ha='center')
fig.text(7, -50, 'vacation', ha='center')


# In[18]:


fig =german_credit_df.Age.hist(bins=60)
fig.text(40, -10, 'Age', ha='center')
fig.text(0, 40, 'Frequency', ha='center')


# In[19]:


fig = german_credit_df["Job"].hist()
fig.text(-0.5, 400, 'Frequency', ha='center')
fig.text(0, -100, 'UnSkilled', ha='center')
fig.text(1, -100, 'UnSkilled Resident', ha='center')
fig.text(2, -100, 'Skilled', ha='center')
fig.text(3, -100, 'Highly Skilled', ha='center')


# # Result:

# 1. People from Age 23 to 32 are the target customer and the amount can be in range 2000 to 5000 <currency>.
# 2. Offers for car loan and radio loan can pick up more customers or lenders.
# 3. Short term credit with credit range 2000 t0 5000 yield maximum customer and profits.

# In[20]:


from sklearn.cluster import KMeans;
from sklearn.decomposition import PCA; 
from sklearn.preprocessing import normalize;
y = KMeans().fit_predict(german_credit_df)
X_norm = normalize(german_credit_df);
y_PCA = PCA(n_components=2).fit_transform(X_norm,2);
y_PCA.shape


# In[22]:


plt.scatter(german_credit_df['Credit amount'],german_credit_df['Age'],c=y)
plt.figure()
plt.scatter(y_PCA[:,0],y_PCA[:,1],c=y)


# In[ ]:




