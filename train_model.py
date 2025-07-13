import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import joblib

# Dummy dataset
df = pd.DataFrame({
    'budget': [10000000, 5000000, 7500000, 20000000],
    'genre': ['Action', 'Drama', 'Comedy', 'Action'],
    'revenue': [100000000, 40000000, 30000000, 120000000]
})

encoder = OneHotEncoder()
genres_encoded = encoder.fit_transform(df[['genre']]).toarray()

X = np.concatenate([df[['budget']].values, genres_encoded], axis=1)
y = df['revenue']

model = RandomForestRegressor()
model.fit(X, y)

joblib.dump(model, 'models/box_office_model.pkl')
joblib.dump(encoder, 'models/genre_encoder.pkl')
print("Model trained and saved.")
