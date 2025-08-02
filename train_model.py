import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Create 'data' directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Example data
data = pd.DataFrame({
    'square_footage': [1000, 1500, 2000, 2500, 3000],
    'bedrooms': [2, 3, 3, 4, 5],
    'bathrooms': [1, 2, 2, 3, 4],
    'price': [3000000, 4500000, 6000000, 7500000, 9000000]
})

X = data[['square_footage', 'bedrooms', 'bathrooms']]
y = data['price']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model on scaled data
model = LinearRegression()
model.fit(X_scaled, y)

# Save model to data/ directory
joblib.dump(model, 'data/linear_model.pkl')

# Save scaler to data/ directory
joblib.dump(scaler, 'data/scaler.pkl')

print("Model and Scaler saved successfully in 'data/' directory.")
