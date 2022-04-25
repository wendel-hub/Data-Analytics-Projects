#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


# In[5]:


df = pd.read_csv(r'Downloads\train_ctrUa4K.csv')
test = pd.read_csv(r'Downloads\test_lAUu6dG.csv')
df_x = df[['Gender','Married','Dependents',"Education",'ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area',"Self_Employed",]]
df_y = df['Loan_Status']
df_x.head()


# Filling Null values with some appropriate values 

# In[6]:


lbl = LabelEncoder()
lists = df_x.select_dtypes(include = 'number').columns.tolist()

lbs = ['Gender','Married',"Education","Self_Employed",'Property_Area']
df_x['Gender'].fillna('Male',inplace = True)
df_x['Married'].fillna('No',inplace = True)
df_x["Education"].fillna('Not Graduate',inplace = True)
df_x["Self_Employed"].fillna('No',inplace = True)
df_x["Property_Area"].fillna('Rural',inplace = True)
df_x['Dependents'].fillna('0',inplace = True)


# Instead of manually doing the filling, this is an easier way: Using a loop to automatically look throughout the values and do the replacement

# In[7]:


for i in lists:
    df_x[i] =df_x[i].fillna(0)

df_x['Dependents'].astype('str')


# Code for doing label encoding on the dataframe. 

# In[8]:


for i in lbs:
    df_x[i] = lbl.fit_transform(df_x[i])

    
df_x['Dependents'] = lbl.fit_transform(df_x['Dependents'])    
df_y = lbl.fit_transform(df_y)


# Training the Data through the Random Forest Classifier

# In[9]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(criterion ='entropy',random_state = 20)
clf.fit(df_x,df_y)


# Doing Feature Engineering for the test data set too

# In[10]:


data_x = test[['Gender','Married','Dependents',"Education",'ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area',"Self_Employed",]]
lists = df_x.select_dtypes(include = 'number').columns.tolist()

lbs = ['Gender','Married',"Education","Self_Employed",'Property_Area']
data_x['Gender'].fillna('Male',inplace = True)
data_x['Married'].fillna('No',inplace = True)
data_x["Education"].fillna('Not Graduate',inplace = True)
data_x["Self_Employed"].fillna('No',inplace = True)
data_x["Property_Area"].fillna('Rural',inplace = True)
data_x['Dependents'].fillna('0',inplace = True)

for i in lists:
    data_x[i] =data_x[i].fillna(0)
    
    
for i in lbs:
    data_x[i] = lbl.fit_transform(data_x[i])

    
data_x['Dependents'] = lbl.fit_transform(data_x['Dependents'])    

predicted = clf.predict(data_x)


# Code to append the Predicted Loan Status to the original Dataframe

# In[119]:


final = pd.DataFrame({'Loan_ID':test['Loan_ID'],'Loan_Status':predicted})

final['New'] = final['Loan_Status'].astype(str)
final['Loan_Status'] = final['New'].replace({'0':'N','1':'Y'})


# Code for extracting the Data as a csv into a specific location

# In[121]:


final.drop('New',axis = 1)
final.to_csv(r'Downloads\submission1.csv',header = True,sep = ',')

