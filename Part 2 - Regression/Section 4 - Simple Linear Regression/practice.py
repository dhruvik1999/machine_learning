import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Salary_Data.csv")
x=dataset.iloc[:,0].values
y=dataset.iloc[:,1].values
x=x.reshape(-1,1)

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=1/3)

from sklearn.linear_model import LinearRegression
linearRegression = LinearRegression()
linearRegression.fit(x_train,y_train)
y_predicts = linearRegression.predict(x_test)

plt.scatter(x_train,y_train,color="red")
plt.plot(x_train, linearRegression.predict(x_train),color="blue")
plt.title("train set Alt")
plt.xlabel("exp")
plt.ylabel("salarey")
plt.show()

plt.scatter(x_test,y_test,color="red")
plt.plot(x_train, linearRegression.predict(x_train),color="blue")
plt.title("train set Alt")
plt.xlabel("exp")
plt.ylabel("salarey")
plt.show()


