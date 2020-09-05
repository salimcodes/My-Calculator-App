import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
%matplotlib inline
salim = pd.read_csv(r"C:\Users\SALIM\Downloads\HeartDiseases.csv")
X= salim.iloc[:,:-1].values
y= salim.iloc[:,13].values
# Encoding categorical data
import sklearn
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:,12]= labelencoder.fit_transform(X[:,12])
#Splitting the dataset into the Training set and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.20)
#print(X_test.shape)
#Scaling dataset
from sklearn import preprocessing
#X_scaled = preprocessing.scale(X_train)
#X_test1 = preprocessing.scale(X_test)
 
#Fitting multiple logistic regression to the training set
from sklearn.ensemble import RandomForestClassifier
s = RandomForestClassifier()
s.fit(X_train, y_train)
y_pred = s.predict(X_train)
score = s.score(X_test,y_test)

import pickle
pickle_out = open('classifier.pkl', 'wb')
pickle_out.close()
