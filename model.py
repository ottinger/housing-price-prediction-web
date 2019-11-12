#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sqlite3
import csv
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pickle

# In[2]:


def read_sales_csv(filename):
    sales = []
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            date = datetime.strptime(row['date'], '%m/%d/%y')
            sales.append({'date': date,
                     'days_ago': (datetime.now() - date).days,
                     'account_number': row['account_number'],
                     'price': int(row['price'])})
    return sales

conn = sqlite3.connect('file:NW_Central_OKC_w_bldg_details.db?mode=ro', uri=True)
c = conn.cursor()

sales = read_sales_csv('sales_list_2019.csv')
#sales = sales + read_sales_csv('sales_list_2018.csv')
#sales = sales + read_sales_csv('sales_list_2017.csv')


# Now let's load the data into DataFrames. We have two tables we're using, one for properties/parcels and one for buildings. So what we'll do is pull out some data about the buildings, sum it up (as some properties have more than 1 building), and add it to the property DF.

# In[3]:


combined_df = pd.read_sql_query("SELECT * FROM realproperty INNER JOIN buildings ON realproperty.id = buildings.local_property_id;", conn)
rp_df = pd.read_sql_query("SELECT * FROM realproperty WHERE property_type = 'Residential'", conn)
rp_df = rp_df.set_index('account_number')

# Get data about the buildings for each parcel. We'll sum up the data for now
# (in cases where there's >1 building on the property)
rp_df['sqft_sum'] = combined_df.groupby(['account_number'])['sq_ft'].sum()
rp_df['bed_sum'] = combined_df.replace(-1,0).groupby(['account_number'])['bedrooms'].sum()
rp_df['bath_sum'] = combined_df.replace(-1,0).groupby(['account_number'])['full_bathrooms'].sum() +    (combined_df.replace(-1,0).groupby(['account_number'])['three_quarters_bathrooms'].sum() * 0.75) +    (combined_df.replace(-1,0).groupby(['account_number'])['half_bathrooms'].sum() * 0.5)
rp_df['room_sum'] = combined_df.replace(-1,0).groupby(['account_number'])['total_rooms'].sum()
rp_df['main_bldg_sqft'] = combined_df.groupby(['account_number'])['sq_ft'].max()
rp_df['year_built'] = combined_df[combined_df.year_built != -1].groupby(['account_number'])['year_built'].mean()
rp_df['bldg_age'] = pd.datetime.now().year - rp_df['year_built']

# Now let's add more features for the main building. To do this we'll create a new DataFrame containing the features of the largest building on each property. While most single-family properties only have one building, we do need to consider that some will have multiple.

# In[4]:


# we want 26584 not 26631
idx = combined_df.groupby(['account_number'])['sq_ft'].transform(max) == combined_df['sq_ft']
pd.set_option('display.max_columns', 60)
pd.set_option('display.max_rows', 100)
combined_df2 = combined_df[idx]


# now that we have the ones with largest sq ft on the parcel, let's weed out duplicates (where there are
# multiple bldgs with identical sq ft)
idx = combined_df2.groupby(['account_number'])['bldg_id'].transform(max) == combined_df2['bldg_id']
combined_df2 = combined_df2[idx]
combined_df2 = combined_df2.set_index(['account_number'])
combined_df2


# Now we have the features for the largest building on each parcel. Let's add some of these to combined_df (like we did above).

# In[5]:


print(len(rp_df))
print(len(combined_df2))
#print(combined_df2.groupby(['account_number'])['bldg_description'])
rp_df['main_bldg_description'] = combined_df2['bldg_description']
rp_df['main_bldg_hvac'] = combined_df2['hvac_type']
rp_df['main_bldg_quality'] = combined_df2['quality_description']
# Floor height is not worth using as it is listed as 8 on the vast majority of residential properties
rp_df['main_bldg_exterior'] = combined_df2['exterior']
rp_df['main_bldg_roof'] = combined_df2['roof_type']
rp_df['main_bldg_year'] = combined_df2['year_built']
rp_df['main_bldg_effective_year'] = combined_df2[['year_built', 'remodel_year']].max(axis=1) # remodel year, else built year
rp_df


# In[6]:


# This filters for subdivision based on number of properties in that sub. Let's change it
# later to filter for number of transactions.
sublist = (rp_df[['subdivision']]
            .assign(count = 0)
            .groupby("subdivision")
            .count())
sublist = sublist.where(sublist['count'] >= 100).dropna()
sublist_names = sublist.index.values
rp_df['subdivision'] = rp_df.apply(lambda x: "Other" if x['subdivision'] not in sublist_names else x['subdivision'], axis=1)


# In[7]:


sales_df = pd.DataFrame(sales)



# In[8]:


merged_df = pd.merge(rp_df, sales_df, on='account_number', how='inner')


# In[9]:


included_fields = ['land_size', 'land_value', 'subdivision', 'bldg_age', 'year_built', 'sqft_sum', 'bed_sum', 'bath_sum', 'price', 'days_ago', 'main_bldg_hvac', 'main_bldg_description', 'main_bldg_exterior', 'main_bldg_quality', 'main_bldg_roof', 'main_bldg_year', 'main_bldg_effective_year', 'room_sum']
new_df = merged_df[included_fields]


# In[10]:


from sklearn.model_selection import train_test_split
from sklearn import preprocessing


# ### One Hot Encoding
#
# We use one hot encoding for the subdivision feature. As a result, we create a feature for each individual neighborhood and other categorical features.

# In[11]:


from sklearn.linear_model import LinearRegression
new_df = (merged_df[included_fields][merged_df.price != 0]
          .dropna()
          .reset_index()
          .drop(columns=['index']))
#new_df = new_df[new_df.price != 0]

ohenc = preprocessing.OneHotEncoder()


# In[12]:


new_df.describe()


# In[13]:


def return_encoded(df, feature_name, feature_prefix):
    ohenc.fit([[x] for x in df[feature_name]])
    encoded = ohenc.transform([[x] for x in df[feature_name]]).toarray()
    return pd.DataFrame(encoded, columns=[feature_prefix+"-" + ohenc.categories_[0][i] for i in range(encoded.shape[1])])


# In[14]:


encoded_df_names = [
    ['subdivision','Sub'],
    ['main_bldg_hvac', 'HVAC'],
    ['main_bldg_description', 'Descr'],
    ['main_bldg_exterior', 'Exterior'],
    ['main_bldg_roof', 'Roof']
]

encoded_dfs = []
for (col_name, prefix) in encoded_df_names:
    cur_df = return_encoded(new_df, col_name, prefix)
    encoded_dfs.append(cur_df)


# In[15]:


unused_col_names = ['main_bldg_quality']

new_df = new_df.drop(columns=([x[0] for x in encoded_df_names]+[x for x in unused_col_names]))
new_df = pd.concat(([new_df] + [x for x in encoded_dfs]), axis=1)


# In[ ]:





# ### Correlation
#
# Let's look at the correlation of the various featues. We see that some features like square footage, # of beds/baths, and land size have a positive correlation.
#
# Some of the more desirable subdivisions have a positive correlation too, while many other neighborhoods have a negative correlation with the sale price (relative to other sales in our data set). Do note that this is correlation for the price of the property - not price relative to building or land size.
#
# Year built/building age is close enough to zero to have no correlation. And as we might expect, days_ago has a negative correlation, indicating that prices have generally been increasing over time.

# In[16]:


pd.set_option('max_rows',100)
dfcorr = new_df.corr()
new_df


# In[17]:


pd.set_option('max_rows',200)


# In[18]:


new_df = new_df.drop('bldg_age',axis=1)


# ### Creating a linear regression model
#
# Let's create our linear regression model.

# In[19]:


X = new_df.drop('price',axis=1)
y = new_df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
linreg = LinearRegression().fit(X_train, y_train)


# In[20]:


from sklearn.metrics import mean_squared_error
import numpy as np

def model_stats(model, Xte, yte, Xtr, ytr, yp):

    print("R^2 for Test Set: " + str(model.score(Xte, yte)))
    print("R^2 for Training Set: " + str(model.score(Xtr,ytr)))
    print("RMS error for Test Set: " + str(np.sqrt(mean_squared_error(yte, yp))))

    print("y_test min: " + str(y_test.min()))
    print("y_test max: " + str(y_test.max()))
    print("y_predict min: " + str(yp.min()))
    print("y_predict max: " + str(yp.max()))


# And if we look at the predicted values, some turn out to be negative. Uh oh.

# In[21]:


'''
import numpy as np
ch_oh = ohenc.transform([["EDGEMERE TERRACE"]]).toarray()
ch_args = [10000, 40000, 1990, 3000, 4, 3, 10] + ch_oh.tolist()[0]
ch_args = [6400, 34790, 1500, 1990, 4, 3, 10] + ch_oh.tolist()[0]
#ch_args = ch_oh.tolist()
linreg.predict([ch_args])
'''

y_predict = linreg.predict(X_test)
model_stats(linreg,X_test,y_test,X_train,y_train,y_predict)


# In[22]:


# ### Bayesian Ridge with Polynomial Features

# In[27]:


from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
# bayesian ridge worked well-ish
# Crashes for PolynomialFeatures degree 3 though. Not sure why...
bayesian_ridge = make_pipeline(PolynomialFeatures(2), BayesianRidge())
bayesian_ridge.fit(X_train, y_train)


# In[28]:


bayesian_ridge.score(X_test, y_test)


# In[29]:


y_predict = bayesian_ridge.predict(X_test)

model_stats(bayesian_ridge,X_test,y_test,X_train,y_train,y_predict)

pickle.dump(bayesian_ridge, open('bayesian_ridge.pkl', 'wb'))
pickle.dump(X.columns, open('column_names.pkl', 'wb'))
