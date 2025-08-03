import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Ensure 'data/' directory exists
os.makedirs("data", exist_ok=True)

# Load dataset
try:
    df = pd.read_csv("data/train.csv")
except FileNotFoundError:
    print("Error: 'data/train.csv' not found.")
    exit()

# Define features and target
features = ['GrLivArea', 'OverallQual', 'TotalBsmtSF']
target = 'SalePrice'

# Drop rows with missing values
df = df[features + [target]].dropna()

# Features and target
X = df[features]
y = df[target]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
score = model.score(X_test, y_test)
print(f"Model trained. R² score: {score:.4f}")

# Save model and scaler in 'data/' folder
with open("data/house_price_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("data/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model and scaler saved to 'data/' directory.")
