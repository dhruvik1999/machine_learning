import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Data.csv")
x=dataset.iloc[:,0:3].values
y=dataset.iloc[:,3].values


from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN",strategy="mean",axis=0)
imputer = imputer.fit(x[:,1:3])
x[:,1:3]=imputer.transform(x[:,1:3])


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelEncoder = LabelEncoder()
y=labelEncoder.fit_transform(y)
x[:,0] =  labelEncoder.fit_transform(x[:,0])

oneHotEncoder = OneHotEncoder(categorical_features=[0])
x=oneHotEncoder.fit_transform(x).toarray()

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0,test_size=0.3)
