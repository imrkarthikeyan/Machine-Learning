import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import pickle

data=pd.read_csv("data.csv")

X=data[['Glucose','BMI','Age']]
y=data['Outcome']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

model=LogisticRegression()

model.fit(X_train,y_train)

predictions=model.predict(X_test)

accuracy=accuracy_score(y_test,predictions)
confusionMatrix=confusion_matrix(y_test,predictions)
classificationReport=classification_report(y_test,predictions)

print("Accuracy : ",accuracy)
print("Confusion Matrix : ",confusionMatrix)
print("Classification Report : ",classificationReport)

pickle.dump(model, open("diabetes_prediction_model", "wb"))