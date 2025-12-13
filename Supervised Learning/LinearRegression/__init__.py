import pickle
import pandas as pd

model=pickle.load(open("house_price_model.pkl","rb"))

area=float(input("Enter house area in sqft : "))
bedrooms=int(input("Enter number of Bed rooms : "))

input_data=pd.DataFrame([[area,bedrooms]],columns=['Area','Bedrooms'])
price=model.predict(input_data)

print("Estimated House Price : ",round(price[0],2),"Lakhs")