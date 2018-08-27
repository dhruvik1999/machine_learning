import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset =pd.read_csv("50_Startups.csv")
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelEncoder=LabelEncoder()
x[:,-1]=labelEncoder.fit_transform(x[:,-1])
oneHotEncoder=OneHotEncoder(categorical_features=[3])
x=oneHotEncoder.fit_transform(x).toarray()

x=x[:,1:]

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
linearRegression=LinearRegression()
linearRegression.fit(x,y)
y_predict=linearRegression.predict(x_test)


plt.scatter(range(50),y,color="red")
plt.plot(range(50),linearRegression.predict(x),color="blue")
plt.show()
