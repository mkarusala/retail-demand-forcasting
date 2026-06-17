import pandas as pd
import joblib

model = joblib.load("models/retail_forecast_model.pkl")

sample_data = pd.DataFrame({
    "store": [1],
    "item": [15],
    "year": [2018],
    "month": [7],
    "day": [1],
    "day_of_week": [0]
})

prediction = model.predict(sample_data)

predicted_sales = round(prediction[0])

safety_stock = 20
recommended_inventory = predicted_sales + safety_stock

current_stock = 80

print("Predicted Sales:", predicted_sales)
print("Safety Stock:", safety_stock)
print("Recommended Inventory:", recommended_inventory)
print("Current Stock:", current_stock)

if current_stock < predicted_sales:
    print("⚠ Stockout Risk!")
else:
    print("✅ Stock Level Safe")