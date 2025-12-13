import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import pickle

data=pd.read_csv("data.csv")

X=data[["Area","Bedrooms"]]
y=data["Price"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

model=LinearRegression()
model.fit(X_train,y_train)

predictions=model.predict(X_test)

error=mean_squared_error(y_test,predictions)
score=r2_score(y_test,predictions)

print("Mean Squared Error : ",error)
print("R2 Score : ",score)

pickle.dump(model,open("house_price_model.pkl","wb"))