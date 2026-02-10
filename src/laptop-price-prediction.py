#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Libraries
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")


# In[2]:


# Loading the data
df = pd.read_csv('../data/Cleaned_Laptop_data.csv')
df


# In[3]:


# Chaecking missing/null data
df.isnull().sum()


# In[4]:


# drop useless feature column
df.drop(columns=['os','os_bit','weight','msoffice'],inplace=True)
df


# In[5]:


df.info()


# In[6]:


# Chaecking missing/null data
df.isnull().sum()


# In[7]:


# 3 null/missing data treatment by mean
df['Price'].fillna(int(df['Price'].mean()), inplace=True) 
df


# In[8]:


# now there is no null/missing data
df.isnull().sum()


# In[9]:


df.columns


# In[10]:


# we can see that there is no column named "Unnamed: 0". So no need to drop any column from the dataset.


# In[11]:


# Duplicate data checking, there is no duplicate data
print(df.duplicated().sum())
print(df.shape)


# In[12]:


# Searching symbol character in dataset
import pandas as pd
import re
symbol_pattern = r'[^\w\s]'
for column in df.columns:
    column_text = ' '.join(df[column].astype(str))
    symbols_found = re.findall(symbol_pattern, column_text)
    if symbols_found:
        print(f"Symbols found in column '{column}': {symbols_found}")
    else:
        print(f"No symbols found in column '{column}'")


# In[13]:


# treatment of symbol character
import pandas as pd
import re
symbol_pattern = r'[^\w\s]'
def remove_symbols(text):
    return re.sub(symbol_pattern, '', str(text))
for column in df.columns:
    df[column] = df[column].apply(remove_symbols)
df


# In[14]:


# Separating categorical and numerical values
catvars = df.select_dtypes(include=['object']).columns
numvars = df.select_dtypes(include = ['int32','int64','float32','float64']).columns
catvars,numvars


# In[15]:


# data presentation/visualization
plt.figure(figsize=(15,7))
sn.distplot(df['Price'],color='red')
plt.show()


# In[16]:


# Label Encoding of catagorical column
import pandas as pd
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
columns_to_encode = ['brand', 'model', 'processor_brand', 'processor_Name',
        'processor_gnrtn', 'ram_gb', 'Apps', 'ssd', 'hdd', 'graphic_card_gb',
        'display_size', 'warranty', 'Touchscreen']
for column in columns_to_encode:
    df[column] = label_encoder.fit_transform(df[column])
df


# In[17]:


# Correlation between target and other column
import seaborn as sns
sns.heatmap(df.corr());


# In[18]:


df.info()


# In[19]:


# we have separted features columns
x = df.drop('Price',axis=1)
x


# In[20]:


# we have separted target columns "Price"
y = df.Price
y


# In[21]:


# Feature Selection & Scoring
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
FEATURES = SelectKBest (score_func= f_classif)
FEATURES.fit(x,y)


# In[22]:


# Featire Scoring
FEATURES.scores_


# In[23]:


# Scoring DataFrame
score_col = pd.DataFrame(data = FEATURES.scores_,columns = ['score'])
score_col


# In[24]:


# Feature DataFrame
Name_col=pd.DataFrame(data = x.columns,columns = ['Features'])
Name_col


# In[25]:


# Feature & Scoring Combine DataFrame
Features_score=pd.concat([Name_col, score_col], axis=1)
Features_score


# In[26]:


# We have taken top 16 scoring features
Features_score.nlargest(16,'score')


# In[27]:


# Splitting the training data and testing data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.20,random_state=2)
x_train.shape
x_test.shape


# In[28]:


# Model Creation by LinearRegression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)


# In[29]:


# Prediction result
y_pred = model.predict(x_test)
y_pred


# In[30]:


# model accuracy
from sklearn.linear_model import LinearRegression as LR
lr = LR()
lr.fit(x_train, y_train)
lr.score(x_test, y_test)


# In[31]:


# Cross Val function
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
model = LinearRegression()
cv_scores = cross_val_score(model, x, y, cv=5)
cv_scores
np.mean(cv_scores)


# In[32]:


# Cross Val Function accuracy
from sklearn.metrics import mean_squared_error
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
mse


# In[ ]:




