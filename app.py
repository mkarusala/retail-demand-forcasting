import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("models/retail_forecast_model.pkl")

# Load dataset
df = pd.read_csv("data/train.csv")
df["date"] = pd.to_datetime(df["date"])
df["month"] = df["date"].dt.month

# Sidebar
st.sidebar.title("Dashboard Menu")
st.sidebar.write("Retail Intelligence System")

st.sidebar.header("Project Statistics")

st.sidebar.metric("Stores", 10)
st.sidebar.metric("Products", 50)
st.sidebar.metric("Records", "913K")
st.sidebar.metric("Model MAE", "6.28")
st.sidebar.success("Model Status: Active")

# Main page
st.title("Retail Demand Forecasting System")
st.subheader("AI-Powered Inventory Optimization Dashboard")
st.caption("Machine Learning Powered Retail Forecasting & Inventory Optimization Platform")

total_sales = int(df["sales"].sum())
avg_sales = round(df["sales"].mean(), 2)

top_item = (
    df.groupby("item")["sales"]
    .sum()
    .idxmax()
)

best_month = (
    df.groupby("month")["sales"]
    .sum()
    .idxmax()
)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric("Total Sales", f"{total_sales:,}")

with kpi2:
    st.metric("Avg Sales", avg_sales)

with kpi3:
    st.metric("Top Item", top_item)

with kpi4:
    st.metric("Best Month", best_month)

st.success("AI Forecasting Engine Running Successfully")

# Inputs
store = st.number_input(
    "Store ID",
    min_value=1,
    max_value=10,
    value=1
)

item = st.number_input(
    "Item ID",
    min_value=1,
    max_value=50,
    value=15
)

month = st.number_input(
    "Month",
    min_value=1,
    max_value=12,
    value=7
)

current_stock = st.number_input(
    "Current Stock",
    min_value=0,
    value=80
)

# Prediction Button
if st.button("Predict Demand"):

    sample_data = pd.DataFrame({
        "store": [store],
        "item": [item],
        "year": [2018],
        "month": [month],
        "day": [1],
        "day_of_week": [0]
    })

    prediction = model.predict(sample_data)

    predicted_sales = round(prediction[0])

    safety_stock = 20
    recommended_inventory = predicted_sales + safety_stock

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Predicted Sales", predicted_sales)

    with col2:
        st.metric("Recommended Inventory", recommended_inventory)

    if current_stock < predicted_sales:
        st.error("⚠ Stockout Risk!")
    else:
        st.success("✅ Stock Level Safe")

    # Sales Analytics
    st.header("Sales Analytics")

    monthly_sales = df.groupby("month")["sales"].sum()

    fig, ax = plt.subplots()

    ax.plot(
        monthly_sales.index,
        monthly_sales.values,
        marker="o"
    )

    ax.set_title("Actual Monthly Sales Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Sales")

    st.pyplot(fig)

    # Top Selling Products

    st.subheader("Top 10 Best Selling Products")

    top_items = (
        df.groupby("item")["sales"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    fig2, ax2 = plt.subplots()

    ax2.bar(
        top_items.index.astype(str),
        top_items.values
    )

    ax2.set_title("Top 10 Best Selling Products")
    ax2.set_xlabel("Item ID")
    ax2.set_ylabel("Total Sales")

    st.pyplot(fig2)

    # Business Insights

    st.header("Business Insights")

    st.info("📈 Peak sales occur during July.")
    st.info("📉 Lowest sales occur during January and February.")
    st.info("🏪 Inventory should be increased before summer months.")
    st.warning("⚠ Stores with stock below predicted demand are at risk of stockouts.")
    st.success("🤖 AI model helps optimize inventory and reduce lost sales.")

st.markdown("---")
st.caption(
    "Built using Python, Scikit-Learn, Pandas, Matplotlib and Streamlit"
)
