import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

<<<<<<< HEAD
df = pd.read_csv("data/train.csv")

df["date"] = pd.to_datetime(df["date"])

=======
# Load data
df = pd.read_csv("data/train.csv")

# Convert date
df["date"] = pd.to_datetime(df["date"])

# Feature Engineering
>>>>>>> cf76ecf812facebe89c46f13f55039267ee1b5a7
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["day_of_week"] = df["date"].dt.dayofweek

<<<<<<< HEAD
X = df[["store", "item", "year", "month", "day", "day_of_week"]]

y = df["sales"]

=======
# Features (X)
X = df[["store", "item", "year", "month", "day", "day_of_week"]]

# Target (y)
y = df["sales"]

# Train/Test Split
>>>>>>> cf76ecf812facebe89c46f13f55039267ee1b5a7
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

<<<<<<< HEAD
=======
# Model
>>>>>>> cf76ecf812facebe89c46f13f55039267ee1b5a7
model = RandomForestRegressor(
    n_estimators=50,
    random_state=42
)

<<<<<<< HEAD
model.fit(X_train, y_train)

predictions = model.predict(X_test)

=======
# Train
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
>>>>>>> cf76ecf812facebe89c46f13f55039267ee1b5a7
mae = mean_absolute_error(y_test, predictions)

print("Mean Absolute Error:", round(mae, 2))

<<<<<<< HEAD
=======
# Feature Importance
>>>>>>> cf76ecf812facebe89c46f13f55039267ee1b5a7
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

print("\nFeature Importance:")
print(importance.sort_values(by="Importance", ascending=False))

import joblib

joblib.dump(model, "models/retail_forecast_model.pkl")

print("\nModel saved successfully!")