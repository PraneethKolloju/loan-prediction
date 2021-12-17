
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import joblib

df1=pd.read_csv('train.csv')
df2=pd.read_csv('test.csv')

df1=df1.drop(['Loan_ID','Gender'],axis=1)
df2=df2.drop(['Loan_ID','Gender'],axis=1)

df1

df1.isnull().sum()

df1['Married']=df1['Married'].fillna('None')

for i in df1:
  if df1[i].dtypes==object:
    df1[i]=df1[i].fillna('None')
  else:
    df1[i]=df1[i].fillna(df1[i].mean())

for i in df2:
  if df2[i].dtypes==object:
    df2[i]=df2[i].fillna('None')
  else:
    df2[i]=df2[i].fillna(df2[i].mean())

df2.isnull().sum()

from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
for i in df1:
  if df1[i].dtypes==object:
      df1[i]=lb.fit_transform(df1[i])
lb.classes_

from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
for i in df2:
  if df2[i].dtypes==object:
      df2[i]=lb.fit_transform(df2[i])

df1=df1.values
df2=df2.values

train=df1[:,:-1]
test=df1[:,-1]



from sklearn.preprocessing import StandardScaler
sctrain=StandardScaler()
sctest=StandardScaler()
train=sctrain.fit_transform(train)
realtest=sctest.fit_transform(df2)

from sklearn.svm import SVC
svc=SVC(kernel='rbf',random_state=0)
svc.fit(train,test)

pickle.dump(svc,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))

