
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.0** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-data-analysis/resources/0dhYG) course resource._
# 
# ---

# # The Series Data Structure

# In[1]:


import pandas as pd
get_ipython().magic('pinfo pd.Series')


# In[2]:


animals = ['Tiger', 'Bear', 'Moose']
pd.Series(animals)


# In[3]:


numbers = [1, 2, 3]
pd.Series(numbers)


# In[4]:


animals = ['Tiger', 'Bear', None]
pd.Series(animals)


# In[5]:


numbers = [1, 2, None]
pd.Series(numbers)


# In[6]:


import numpy as np
np.nan == None


# In[7]:


np.nan == np.nan


# In[8]:


np.isnan(np.nan)


# In[9]:


sports = {'Archery': 'Bhutan',
          'Golf': 'Scotland',
          'Sumo': 'Japan',
          'Taekwondo': 'South Korea'}
s = pd.Series(sports)
s


# In[10]:


s.index


# In[11]:


s = pd.Series(['Tiger', 'Bear', 'Moose'], index=['India', 'America', 'Canada'])
s


# In[12]:


sports = {'Archery': 'Bhutan',
          'Golf': 'Scotland',
          'Sumo': 'Japan',
          'Taekwondo': 'South Korea'}
s = pd.Series(sports, index=['Golf', 'Sumo', 'Hockey'])
s


# # Querying a Series

# In[13]:


sports = {'Archery': 'Bhutan',
          'Golf': 'Scotland',
          'Sumo': 'Japan',
          'Taekwondo': 'South Korea'}
s = pd.Series(sports)
s


# In[14]:


s.iloc[3]


# In[15]:


s.loc['Golf']


# In[16]:


s[3]


# In[17]:


s['Golf']


# In[18]:


sports = {99: 'Bhutan',
          100: 'Scotland',
          101: 'Japan',
          102: 'South Korea'}
s = pd.Series(sports)


# In[23]:


s[0] #This won't call s.iloc[0] as one might expect, it generates an error instead


# In[19]:


s = pd.Series([100.00, 120.00, 101.00, 3.00])
s


# In[20]:


total = 0
for item in s:
    total+=item
print(total)


# In[21]:


import numpy as np

total = np.sum(s)
print(total)


# In[22]:


#this creates a big series of random numbers
s = pd.Series(np.random.randint(0,1000,10000))
s.head()


# In[23]:


len(s)


# In[24]:


get_ipython().run_cell_magic('timeit', '-n 100', 'summary = 0\nfor item in s:\n    summary+=item')


# In[25]:


get_ipython().run_cell_magic('timeit', '-n 100', 'summary = np.sum(s)')


# In[26]:


s+=2 #adds two to each item in s using broadcasting
s.head()


# In[ ]:


for label, value in s.iteritems():
    s.set_value(label, value+2)
s.head()


# In[ ]:


get_ipython().run_cell_magic('timeit', '-n 10', 's = pd.Series(np.random.randint(0,1000,10000))\nfor label, value in s.iteritems():\n    s.loc[label]= value+2')


# In[ ]:


get_ipython().run_cell_magic('timeit', '-n 10', 's = pd.Series(np.random.randint(0,1000,10000))\ns+=2')


# In[ ]:


s = pd.Series([1, 2, 3])
s.loc['Animal'] = 'Bears'
s


# In[4]:


original_sports = pd.Series({'Archery': 'Bhutan',
                             'Golf': 'Scotland',
                             'Sumo': 'Japan',
                             'Taekwondo': 'South Korea'})
cricket_loving_countries = pd.Series(['Australia',
                                      'Barbados',
                                      'Pakistan',
                                      'England'], 
                                   index=['Cricket',
                                          'Cricket',
                                          'Cricket',
                                          'Cricket'])
all_countries = original_sports.append(cricket_loving_countries)


# In[5]:


original_sports


# In[6]:


cricket_loving_countries


# In[7]:


all_countries


# In[8]:


all_countries.loc['Cricket']


# # The DataFrame Data Structure

# In[1]:


import pandas as pd
purchase_1 = pd.Series({'Name': 'Chris',
                        'Item Purchased': 'Dog Food',
                        'Cost': 22.50})
purchase_2 = pd.Series({'Name': 'Kevyn',
                        'Item Purchased': 'Kitty Litter',
                        'Cost': 2.50})
purchase_3 = pd.Series({'Name': 'Vinod',
                        'Item Purchased': 'Bird Seed',
                        'Cost': 5.00})
df = pd.DataFrame([purchase_1, purchase_2, purchase_3], index=['Store 1', 'Store 1', 'Store 2'])
df.head()


# In[2]:


df.loc['Store 2']


# In[3]:


type(df.loc['Store 2'])


# In[4]:


df.loc['Store 1']


# In[5]:


df.loc['Store 1', 'Cost']


# In[6]:


df.T


# In[7]:


df.T.loc['Cost']


# In[8]:


df['Cost']


# In[9]:


df.loc['Store 1']['Cost']


# In[10]:


df.loc[:,['Name', 'Cost']]


# In[11]:


df.drop('Store 1')


# In[12]:


df


# In[13]:


copy_df = df.copy()
copy_df = copy_df.drop('Store 1')
copy_df


# In[14]:


get_ipython().magic('pinfo copy_df.drop')


# In[15]:


del copy_df['Name']
copy_df


# In[16]:


df['Location'] = None
df


# # Dataframe Indexing and Loading

# In[18]:


costs = df['Cost']
costs


# In[19]:


costs+=2
costs


# In[20]:


df


# In[21]:


get_ipython().system('cat olympics.csv')


# In[22]:


df = pd.read_csv('olympics.csv')
df.head()


# In[23]:


df = pd.read_csv('olympics.csv', index_col = 0, skiprows=1)
df.head()


# In[24]:


df.columns


# In[25]:


for col in df.columns:
    if col[:2]=='01':
        df.rename(columns={col:'Gold' + col[4:]}, inplace=True)
    if col[:2]=='02':
        df.rename(columns={col:'Silver' + col[4:]}, inplace=True)
    if col[:2]=='03':
        df.rename(columns={col:'Bronze' + col[4:]}, inplace=True)
    if col[:1]=='№':
        df.rename(columns={col:'#' + col[1:]}, inplace=True) 

df.head()


# # Querying a DataFrame

# In[26]:


df['Gold'] > 0


# In[27]:


only_gold = df.where(df['Gold'] > 0)
only_gold.head()


# In[28]:


only_gold['Gold'].count()


# In[29]:


df['Gold'].count()


# In[30]:


only_gold = only_gold.dropna()
only_gold.head()


# In[ ]:


only_gold = df[df['Gold'] > 0]
only_gold.head()


# In[31]:


len(df[(df['Gold'] > 0) | (df['Gold.1'] > 0)])


# In[32]:


df[(df['Gold.1'] > 0) & (df['Gold'] == 0)]


# # Indexing Dataframes

# In[33]:


df.head()


# In[34]:


df['country'] = df.index
df = df.set_index('Gold')
df.head()


# In[36]:


df = df.reset_index()
df.head()


# In[1]:


df = pd.read_csv('census.csv')
df.head(100)


# In[38]:


df['SUMLEV'].unique()


# In[40]:


df=df[df['SUMLEV'] == 50]
df.head()


# In[41]:


columns_to_keep = ['STNAME',
                   'CTYNAME',
                   'BIRTHS2010',
                   'BIRTHS2011',
                   'BIRTHS2012',
                   'BIRTHS2013',
                   'BIRTHS2014',
                   'BIRTHS2015',
                   'POPESTIMATE2010',
                   'POPESTIMATE2011',
                   'POPESTIMATE2012',
                   'POPESTIMATE2013',
                   'POPESTIMATE2014',
                   'POPESTIMATE2015']
df = df[columns_to_keep]
df.head()


# In[42]:


df = df.set_index(['STNAME', 'CTYNAME'])
df.head()


# In[43]:


df.loc['Michigan', 'Washtenaw County']


# In[44]:


df.loc[ [('Michigan', 'Washtenaw County'),
         ('Michigan', 'Wayne County')] ]


# # Missing values

# In[ ]:


df = pd.read_csv('log.csv')
df


# In[50]:


get_ipython().magic('pinfo df.fillna')


# In[47]:


df = df.set_index('time')
df = df.sort_index()
df


# In[51]:


df = df.reset_index()
df = df.set_index(['time', 'user'])
df


# In[53]:


df = df.fillna(method='ffill')
df.head()


# In[54]:


df


# In[ ]:




