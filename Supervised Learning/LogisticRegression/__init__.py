import pickle
import pandas as pd

model=pickle.load(open("diabetes_prediction_model", "rb"))

glucose=int(input("Enter Glucose level : "))
bmi=float(input("Enter BMI value : "))
age=int(input("Enter age : "))

input_data=pd.DataFrame(
    [[glucose,bmi,age]],
    columns=["Glucose","BMI","Age"]
)

result=model.predict(input_data)[0]
probability=model.predict_proba(input_data)[0][1]

if result==1:
    print(f"Result : You have diabetes (Probability : {probability:.6f})")
else:
    print(f"Result : You don't have diabetes (Probability : {probability:.6f})")

