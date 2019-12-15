import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0  )
"""
from sklearn.preprocessing import StandardScaler
sc_t = StandardScaler()
x_train = sc_t.fit_transform(x_train)
x_test = sc_t.transform(x_test)
"""





"""
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer = imputer.fit(x[:,1:3]) 
x[:,1:3] = imputer.transform(x[:,1:3])
"""
"""
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelEncoder_x = LabelEncoder()
x[:,0] = labelEncoder_x.fit_transform(x[:,0])
oneHotEncoder =  OneHotEncoder(categorical_features=[0])
x = oneHotEncoder.fit_transform(x).toarray()
labelEncoder = LabelEncoder()
y = labelEncoder.fit_transform(y)
"""