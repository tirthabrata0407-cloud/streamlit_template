# pip install streamlit prophet openpyxl scikit-learn matplotlib pandas numpy
# Complete RetailPulse Streamlit Dashboard Code
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------
st.set_page_config(
    page_title="RetailPulse Dashboard",
    layout="wide"
)

# -------------------------------------------------
# TITLE
# -------------------------------------------------
st.title("📊 RetailPulse - AI Powered Retail Analytics Dashboard")

st.markdown("---")

# -------------------------------------------------
# LOAD DATASET
# -------------------------------------------------
@st.cache_data
def load_data():
    try:
        # Try to load the data file
        if os.path.exists("merged_cleaned_retail_data.xlsx"):
            df = pd.read_excel("merged_cleaned_retail_data.xlsx")
            return df
        else:
            st.error("❌ Data file 'merged_cleaned_retail_data.xlsx' not found. Please upload it to the repository.")
            return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()

if df is None:
    st.stop()

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
menu = st.sidebar.selectbox(
    "Select Analysis",
    [
        "Dataset Overview",
        "Data Cleaning",
        "Feature Engineering",
        "EDA",
        "Customer Segmentation",
        "Demand Forecasting",
        "Churn Prediction"
    ]
)

# =================================================
# 1. DATASET OVERVIEW
# =================================================
if menu == "Dataset Overview":

    st.header("📁 Dataset Overview")

    st.subheader("First 5 Rows")
    st.dataframe(df.head())

    st.subheader("Dataset Shape")
    st.write(df.shape)

    st.subheader("Column Names")
    st.write(df.columns.tolist())

    st.subheader("Data Types")
    st.write(df.dtypes)

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

# =================================================
# 2. DATA CLEANING
# =================================================
elif menu == "Data Cleaning":

    st.header("🧹 Data Cleaning")

    st.subheader("Original Shape")
    st.write(df.shape)

    # Remove missing values
    df_cleaned = df.dropna()

    # Remove duplicates
    df_cleaned = df_cleaned.drop_duplicates()

    # Remove negative quantity if column exists
    if 'Quantity' in df_cleaned.columns:
        df_cleaned = df_cleaned[df_cleaned['Quantity'] > 0]

    # Remove negative price if column exists
    if 'Price' in df_cleaned.columns:
        df_cleaned = df_cleaned[df_cleaned['Price'] > 0]

    st.subheader("Cleaned Dataset Shape")
    st.write(df_cleaned.shape)

    st.subheader("Missing Values After Cleaning")
    st.write(df_cleaned.isnull().sum())

    st.success("✅ Data Cleaning Completed Successfully")

# =================================================
# 3. FEATURE ENGINEERING
# =================================================
elif menu == "Feature Engineering":

    st.header("⚙️ Feature Engineering")

    df_feat = df.copy()
    
    # Find date column
    date_col = None
    for col in df_feat.columns:
        if 'date' in col.lower() or 'invoice' in col.lower():
            if df_feat[col].dtype == 'object':
                try:
                    df_feat[col] = pd.to_datetime(df_feat[col])
                    date_col = col
                    break
                except:
                    continue

    if date_col:
        # Date Features
        df_feat['Year'] = df_feat[date_col].dt.year
        df_feat['Month'] = df_feat[date_col].dt.month
        df_feat['Day'] = df_feat[date_col].dt.day
        df_feat['Weekday'] = df_feat[date_col].dt.day_name()

        st.subheader("Date Features Added")
        st.dataframe(df_feat[[date_col, 'Year', 'Month', 'Day', 'Weekday']].head())
    else:
        st.warning("⚠️ No date column found in the dataset")

    # Total Revenue if column exists
    if 'Total_Amount' in df_feat.columns:
        st.subheader("Total Revenue")
        st.metric("Revenue", f"${round(df_feat['Total_Amount'].sum(), 2):,}")

    # Total Profit if column exists
    if 'Profit' in df_feat.columns:
        st.subheader("Total Profit")
        st.metric("Profit", f"${round(df_feat['Profit'].sum(), 2):,}")

# =================================================
# 4. EDA
# =================================================
elif menu == "EDA":

    st.header("📈 Exploratory Data Analysis")

    df_eda = df.copy()
    
    # Find date and numeric columns
    date_col = None
    for col in df_eda.columns:
        if 'date' in col.lower() or 'invoice' in col.lower():
            if df_eda[col].dtype == 'object':
                try:
                    df_eda[col] = pd.to_datetime(df_eda[col])
                    date_col = col
                    break
                except:
                    continue

    # ---------------------------------------------
    # Monthly Sales
    # ---------------------------------------------
    if date_col and 'Total_Amount' in df_eda.columns:
        st.subheader("Monthly Sales Trend")
        df_eda['Month'] = df_eda[date_col].dt.to_period('M')
        monthly_sales = df_eda.groupby('Month')['Total_Amount'].sum()

        fig, ax = plt.subplots(figsize=(10, 5))
        monthly_sales.plot(kind='line', marker='o', ax=ax)
        ax.set_title("Monthly Sales Trend")
        ax.set_xlabel("Month")
        ax.set_ylabel("Revenue")
        plt.tight_layout()
        st.pyplot(fig)

    # ---------------------------------------------
    # Top Products
    # ---------------------------------------------
    if 'Description' in df_eda.columns and 'Quantity' in df_eda.columns:
        st.subheader("Top 10 Selling Products")
        top_products = df_eda.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(12, 6))
        top_products.plot(kind='bar', ax=ax)
        ax.set_title("Top Selling Products")
        ax.set_xlabel("Products")
        ax.set_ylabel("Quantity")
        plt.xticks(rotation=90)
        plt.tight_layout()
        st.pyplot(fig)

    # ---------------------------------------------
    # Country Sales
    # ---------------------------------------------
    if 'Country' in df_eda.columns and 'Total_Amount' in df_eda.columns:
        st.subheader("Top Countries by Revenue")
        country_sales = df_eda.groupby('Country')['Total_Amount'].sum().sort_values(ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(10, 5))
        country_sales.plot(kind='bar', ax=ax)
        ax.set_title("Top Countries")
        ax.set_xlabel("Country")
        ax.set_ylabel("Revenue")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

    # ---------------------------------------------
    # Customer Type Analysis
    # ---------------------------------------------
    if 'Customer_Type' in df_eda.columns and 'Total_Amount' in df_eda.columns:
        st.subheader("Customer Type Revenue")
        customer_type = df_eda.groupby('Customer_Type')['Total_Amount'].sum()

        fig, ax = plt.subplots()
        customer_type.plot(kind='pie', autopct='%1.1f%%', ax=ax)
        ax.set_ylabel("")
        plt.tight_layout()
        st.pyplot(fig)

# =================================================
# 5. CUSTOMER SEGMENTATION
# =================================================
elif menu == "Customer Segmentation":

    st.header("👥 Customer Segmentation (RFM Analysis)")

    df_seg = df.copy()
    
    # Find date column
    date_col = None
    for col in df_seg.columns:
        if 'date' in col.lower() or 'invoice' in col.lower():
            if df_seg[col].dtype == 'object':
                try:
                    df_seg[col] = pd.to_datetime(df_seg[col])
                    date_col = col
                    break
                except:
                    continue

    if date_col and 'Customer ID' in df_seg.columns:
        snapshot_date = df_seg[date_col].max() + pd.Timedelta(days=1)

        rfm = df_seg.groupby('Customer ID').agg({
            date_col: lambda x: (snapshot_date - x.max()).days,
            'Invoice': 'nunique' if 'Invoice' in df_seg.columns else 'count',
            'Total_Amount': 'sum' if 'Total_Amount' in df_seg.columns else 'count'
        })

        rfm.columns = ['Recency', 'Frequency', 'Monetary']

        st.subheader("RFM Table")
        st.dataframe(rfm.head())

        # Scaling
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm)

        # KMeans
        kmeans = KMeans(n_clusters=4, random_state=42)
        rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

        st.subheader("Cluster Summary")
        st.write(rfm.groupby('Cluster').mean())

        # Visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(rfm['Frequency'], rfm['Monetary'], c=rfm['Cluster'], cmap='viridis')
        ax.set_title("Customer Segmentation")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Monetary")
        plt.colorbar(scatter, ax=ax, label='Cluster')
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("⚠️ Required columns for RFM analysis not found")

# =================================================
# 6. DEMAND FORECASTING
# =================================================
elif menu == "Demand Forecasting":

    st.header("📉 Demand Forecasting")

    df_forecast = df.copy()
    
    # Find date and amount columns
    date_col = None
    amount_col = None
    
    for col in df_forecast.columns:
        if 'date' in col.lower() or 'invoice' in col.lower():
            if df_forecast[col].dtype == 'object':
                try:
                    df_forecast[col] = pd.to_datetime(df_forecast[col])
                    date_col = col
                    break
                except:
                    continue

    for col in df_forecast.columns:
        if 'amount' in col.lower() or 'revenue' in col.lower() or 'sales' in col.lower():
            amount_col = col
            break

    if date_col and amount_col:
        forecast_data = df_forecast.groupby(date_col)[amount_col].sum().reset_index()
        forecast_data.columns = ['ds', 'y']

        st.subheader("Forecast Data")
        st.dataframe(forecast_data.head())

        try:
            # Prophet Model
            model = Prophet(interval_width=0.95)
            model.fit(forecast_data)
            future = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)

            st.subheader("Forecast Results")
            st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

            # Forecast Plot
            fig1 = model.plot(forecast)
            plt.tight_layout()
            st.pyplot(fig1)
        except Exception as e:
            st.error(f"Error in forecasting: {e}")
    else:
        st.warning("⚠️ Required columns for forecasting not found")

# =================================================
# 7. CHURN PREDICTION
# =================================================
elif menu == "Churn Prediction":

    st.header("⚠️ Churn Prediction")

    df_churn = df.copy()
    
    if 'Customer ID' in df_churn.columns:
        customer_data = df_churn.groupby('Customer ID').agg({
            'Total_Amount': 'sum' if 'Total_Amount' in df_churn.columns else 'count',
            'Invoice': 'nunique' if 'Invoice' in df_churn.columns else 'count',
            'Churn': 'max' if 'Churn' in df_churn.columns else 'count'
        }).reset_index()

        customer_data.columns = ['Customer_ID', 'Total_Spent', 'Total_Orders', 'Churn']

        X = customer_data[['Total_Spent', 'Total_Orders']]
        y = customer_data['Churn']

        if len(X) > 0 and y.nunique() > 1:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)

            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)

            st.subheader("Model Accuracy")
            st.metric("Accuracy", f"{round(accuracy * 100, 2)}%")

            # Churn Distribution
            st.subheader("Churn Distribution")
            churn_counts = customer_data['Churn'].value_counts()

            fig, ax = plt.subplots()
            churn_counts.plot(kind='bar', ax=ax, color=['green', 'red'])
            ax.set_title("Customer Churn Distribution")
            ax.set_xlabel("Churn")
            ax.set_ylabel("Count")
            plt.tight_layout()
            st.pyplot(fig)

            st.success("✅ Churn Prediction Completed")
        else:
            st.warning("⚠️ Insufficient data for churn prediction")
    else:
        st.warning("⚠️ Customer ID column not found")
