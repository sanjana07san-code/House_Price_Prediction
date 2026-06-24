import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv("dataset/house_price.csv")

encoder = LabelEncoder()
df['Location'] = encoder.fit_transform(df['Location'])

X = df[['Location','Area','Bedrooms','Bathrooms','Floors','Parking','Age']]
y = df['Price']

model = LinearRegression()

model.fit(X,y)

joblib.dump(model,"model/model.pkl")
joblib.dump(encoder,"model/scaler.pkl")

print("Model Saved Successfully")