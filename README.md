RETAIL DEMAND FORECASTING AND INVENTORY OPTIMIZATION

Overview:

Managing inventory efficiently is a major challenge in retail. Overstocking increases storage costs, while understocking can lead to lost sales and unhappy customers.

In this project, I built a machine learning model to forecast product demand using historical retail sales data. The goal is to help retailers make better inventory decisions by predicting future sales patterns and identifying demand trends.

Objective:

The objective of this project is to predict future product demand using historical sales data and provide insights that can help retailers optimize inventory levels and reduce stock-related issues.

Dataset

The dataset contains approximately 913,000 sales records collected across:

- 10 retail stores
- 50 different products

Each record includes information such as the store, item, date, and number of units sold.

Tools and Libraries:

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn
- Random Forest Regressor

Exploratory Data Analysis:

Before building the model, I performed exploratory data analysis to better understand the dataset.

Observations:

- The dataset contained no missing values.
- Average daily sales were around 52 units.
- Sales showed clear seasonal trends throughout the year.
- July recorded the highest overall sales volume.
- Certain products consistently outperformed others across multiple stores.

These insights helped in selecting relevant features for forecasting.

Feature Engineering:

To capture time-based sales patterns, I extracted several features from the date column:

- Year
- Month
- Day
- Day of Week

These features were combined with store and product information to train the forecasting model.

Model Development:

For demand prediction, I used a Random Forest Regressor, which performs well on structured tabular datasets and can capture non-linear relationships between features.

Model Performance:

Mean Absolute Error (MAE): 6.28

This means that, on average, the model's predictions differ from actual sales by approximately 6 units.

Feature Importance:

1. Item (56.5%)
2. Store (16.0%)
3. Month (12.5%)
4. Day of Week (6.4%)
5. Year (4.9%)
6. Day (3.6%)

The results indicate that product-specific characteristics and store location have the strongest impact on sales demand.

Results:

The model successfully captured sales trends and seasonal behavior across different products and stores. These forecasts can be used as a foundation for inventory planning, demand estimation, and stock management decisions.

Some improvements I plan to explore in the future include:

1. Implementing XGBoost for improved forecasting accuracy
2. Building an inventory optimization module
3. Developing a stockout prediction system
4. Creating an interactive Streamlit dashboard
5. Deploying a user-friendly sales forecasting interface

Conclusion:

This project demonstrates how machine learning can be applied to retail sales data to generate demand forecasts and support inventory management. It also provided valuable experience in data analysis, feature engineering, model development, and performance evaluation.
