import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

# Load the dataset
data = pd.read_csv('data/house_data.csv')

# Features and target
X = data[['sq_footage', 'bedrooms', 'bathrooms']]
y = data['price']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save using joblib
os.makedirs('data', exist_ok=True)
joblib.dump(model, 'data/house_price_model.pkl')

print("Model trained and saved with joblib as 'data/house_price_model.pkl'")
