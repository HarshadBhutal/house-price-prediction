import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing(as_frame=True)
df = housing.frame

print(df.head())

sns.scatterplot(x=df['MedInc'],y=df['MedHouseVal'])
plt.xlabel("median income ")
plt.ylabel("House value ")
plt.show()

x=df.drop("MedHouseVal",axis=1)
y=df["MedHouseVal"]


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

scaler=StandardScaler()
x_norm=scaler.fit_transform(x_train)
x_test_norm=scaler.transform(x_test)

sgd=SGDRegressor(max_iter=1,warm_start=True,tol=None,random_state=42)


iteration=1000
mse_list=[]
r2_list=[]

for i in range(iteration):
   sgd.partial_fit(x_norm,y_train)
   y_pred=sgd.predict(x_test_norm)
   mse=mean_squared_error(y_test,y_pred)
   r2Score=r2_score(y_test,y_pred)
   mse_list.append(mse)
   r2_list.append(r2Score)



plt.scatter(y_test,y_pred)
plt.xlabel("actual value")
plt.ylabel("predicted value")
plt.title("actual value vs predicted value")
plt.show()

plt.plot(range(1,iteration+1),mse_list)
plt.xlabel("iterations")
plt.ylabel("mse")
plt.title("iterations vs mse ")
plt.grid(True)
plt.show()

print("mse ",np.mean(mse_list))
print("r2 score is ", np.mean(r2Score))
print("iterations ",sgd.n_iter_)