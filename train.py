import pandas as pd
import pickle
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Create models folder
os.makedirs("models", exist_ok=True)

# Load dataset
df = pd.read_csv("crop_data.csv")  # <-- your file name from screenshot

# Features & label
X = df.drop("label", axis=1)
y = df["label"]

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train model
model = RandomForestClassifier()
model.fit(X, y_encoded)

# Save files
pickle.dump(model, open("models/model.pkl", "wb"))
pickle.dump(le, open("models/label_encoder.pkl", "wb"))

print("✅ NEW model.pkl created successfully!")