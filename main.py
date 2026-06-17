import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load data
df = pd.read_csv("data/train.csv")

# Use only 50,000 rows
df = df.sample(n=50000, random_state=42)

# Convert date
df["date"] = pd.to_datetime(df["date"])

# Feature Engineering
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["day_of_week"] = df["date"].dt.dayofweek

# Features (X)
X = df[["store", "item", "year", "month", "day", "day_of_week"]]

# Target (y)
y = df["sales"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestRegressor(
    n_estimators=10,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

# Train
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, predictions)

print("Mean Absolute Error:", round(mae, 2))

# Feature Importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

print("\nFeature Importance:")
print(importance.sort_values(by="Importance", ascending=False))

import joblib

joblib.dump(model, "models/retail_forecast_model.pkl")

print("\nModel saved successfully!")