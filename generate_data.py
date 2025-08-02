import pandas as pd
import random

data = {
    'sq_footage': [random.randint(500, 3000) for _ in range(100)],
    'bedrooms': [random.randint(1, 5) for _ in range(100)],
    'bathrooms': [random.randint(1, 3) for _ in range(100)],
}

# Create target variable 'price'
data['price'] = [
    50 * data['sq_footage'][i] + 10000 * data['bedrooms'][i] + 7000 * data['bathrooms'][i] + random.randint(-10000, 10000)
    for i in range(100)
]

df = pd.DataFrame(data)
df.to_csv('data/house_data.csv', index=False)

print("Dataset created successfully.")
