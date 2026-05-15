# pip install -r requirements.txt
# Complete RetailPulse Streamlit Dashboard Code
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Try importing Prophet, show warning if not available
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    st.warning("Prophet not installed. Demand Forecasting will be disabled.")

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
        df = pd.read_excel("merged_cleaned_retail_data.xlsx")
        return df
    except FileNotFoundError:
        st.error("⚠️ Error: 'merged_cleaned_retail_data.xlsx' not found!")
        st.info("Please upload the Excel file to the same directory as app.py")
        return None
    except Exception as e:
        st.error(f"⚠️ Error loading file: {str(e)}")
        return None

df = load_data()

# Stop execution if data couldn't be loaded
if df is None:
    st.stop()

# -------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------
def get_date_column():
    """Dynamically find date column"""
    date_cols = ['Invoice Date', 'Date', 'invoice_date', 'date', 'InvoiceDate']
    for col in date_cols:
        if col in df.columns:
            return col
    return None

def get_column_safe(col_name, alternatives=None):
    """Safely get column, checking alternatives"""
    if col_name in df.columns:
        return col_name
    if alternatives:
        for alt in alternatives:
            if alt in df.columns:
                return alt
    return None

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
menu_options = [
    "Dataset Overview",
    "Data Cleaning",
    "Feature Engineering",
    "EDA",
    "Customer Segmentation",
    "Churn Prediction"
]

if PROPHET_AVAILABLE:
    menu_options.insert(5, "Demand Forecasting")

menu = st.sidebar.selectbox("Select Analysis", menu_options)

# =================================================
# 1. DATASET OVERVIEW
# =================================================
if menu == "Dataset Overview":

    st.header("📁 Dataset Overview")

    st.subheader("First 5 Rows")
    st.dataframe(df.head())

    st.subheader("Dataset Shape")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])

    st.subheader("Column Names")
    st.write(df.columns.tolist())

    st.subheader("Data Types")
    st.write(df.dtypes)

    st.subheader("Missing Values")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        st.write(missing[missing > 0])
    else:
        st.success("✅ No missing values!")

# =================================================
# 2. DATA CLEANING
# =================================================
elif menu == "Data Cleaning":

    st.header("🧹 Data Cleaning")

    st.subheader("Original Shape")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])

    # Remove missing values
    df_cleaned = df.dropna()

    # Remove duplicates
    df_cleaned = df_cleaned.drop_duplicates()

    # Remove negative quantity if column exists
    qty_col = get_column_safe('Quantity', ['quantity', 'Qty', 'qty'])
    if qty_col:
        df_cleaned = df_cleaned[df_cleaned[qty_col] > 0]

    # Remove negative price if column exists
    price_col = get_column_safe('Price', ['price', 'Unit_Price', 'unit_price'])
    if price_col:
        df_cleaned = df_cleaned[df_cleaned[price_col] > 0]

    st.subheader("Cleaned Dataset Shape")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Rows", df_cleaned.shape[0])
    with col2:
        st.metric("Columns", df_cleaned.shape[1])

    st.subheader("Missing Values After Cleaning")
    missing = df_cleaned.isnull().sum()
    if missing.sum() > 0:
        st.write(missing[missing > 0])
    else:
        st.success("✅ No missing values!")

    st.success("✅ Data Cleaning Completed Successfully")

# =================================================
# 3. FEATURE ENGINEERING
# =================================================
elif menu == "Feature Engineering":

    st.header("⚙️ Feature Engineering")

    date_col = get_date_column()
    
    if date_col:
        df['Invoice Date'] = pd.to_datetime(df[date_col], errors='coerce')

        # Date Features
        df['Year'] = df['Invoice Date'].dt.year
        df['Month'] = df['Invoice Date'].dt.month
        df['Day'] = df['Invoice Date'].dt.day
        df['Weekday'] = df['Invoice Date'].dt.day_name()

        st.subheader("Date Features Added")
        st.dataframe(
            df[['Invoice Date', 'Year', 'Month', 'Day', 'Weekday']].head()
        )
    else:
        st.warning("⚠️ No date column found in dataset")

    # Total Revenue
    amount_col = get_column_safe('Total_Amount', ['total_amount', 'Amount', 'amount', 'Total'])
    if amount_col:
        st.subheader("Financial Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Revenue", f"${df[amount_col].sum():,.2f}")
        
        # Profit if available
        profit_col = get_column_safe('Profit', ['profit', 'Net_Profit'])
        if profit_col:
            with col2:
                st.metric("Total Profit", f"${df[profit_col].sum():,.2f}")

# =================================================
# 4. EDA
# =================================================
elif menu == "EDA":

    st.header("📈 Exploratory Data Analysis")

    date_col = get_date_column()
    amount_col = get_column_safe('Total_Amount', ['total_amount', 'Amount', 'amount'])
    qty_col = get_column_safe('Quantity', ['quantity', 'Qty'])
    country_col = get_column_safe('Country', ['country', 'Country_Name'])
    cust_type_col = get_column_safe('Customer_Type', ['customer_type', 'CustomerType'])

    # Monthly Sales
    if date_col and amount_col:
        st.subheader("Monthly Sales Trend")
        df['Invoice Date'] = pd.to_datetime(df[date_col], errors='coerce')
        df['Month'] = df['Invoice Date'].dt.to_period('M')
        monthly_sales = df.groupby('Month')[amount_col].sum()

        fig, ax = plt.subplots(figsize=(12, 5))
        monthly_sales.astype(float).plot(kind='line', marker='o', ax=ax, linewidth=2)
        ax.set_title("Monthly Sales Trend", fontsize=14, fontweight='bold')
        ax.set_xlabel("Month")
        ax.set_ylabel("Revenue ($)")
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # Top Products
    if qty_col and 'Description' in df.columns:
        st.subheader("Top 10 Selling Products")
        top_products = df.groupby('Description')[qty_col].sum() \
                         .sort_values(ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(12, 6))
        top_products.plot(kind='barh', ax=ax, color='steelblue')
        ax.set_title("Top Selling Products", fontsize=14, fontweight='bold')
        ax.set_xlabel("Quantity Sold")
        st.pyplot(fig)

    # Country Sales
    if country_col and amount_col:
        st.subheader("Top Countries by Revenue")
        country_sales = df.groupby(country_col)[amount_col].sum() \
                          .sort_values(ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(12, 6))
        country_sales.plot(kind='bar', ax=ax, color='coral')
        ax.set_title("Top Countries by Revenue", fontsize=14, fontweight='bold')
        ax.set_xlabel("Country")
        ax.set_ylabel("Revenue ($)")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # Customer Type Analysis
    if cust_type_col and amount_col:
        st.subheader("Revenue by Customer Type")
        customer_type = df.groupby(cust_type_col)[amount_col].sum()

        fig, ax = plt.subplots(figsize=(8, 6))
        customer_type.plot(kind='pie', autopct='%1.1f%%', ax=ax, startangle=90)
        ax.set_ylabel("")
        ax.set_title("Revenue Distribution by Customer Type", fontsize=14, fontweight='bold')
        st.pyplot(fig)

# =================================================
# 5. CUSTOMER SEGMENTATION
# =================================================
elif menu == "Customer Segmentation":

    st.header("👥 Customer Segmentation (RFM Analysis)")

    date_col = get_date_column()
    amount_col = get_column_safe('Total_Amount', ['total_amount', 'Amount', 'amount'])
    cust_id_col = get_column_safe('Customer ID', ['customer_id', 'CustomerID', 'Customer_ID'])
    invoice_col = get_column_safe('Invoice', ['invoice', 'Invoice_ID', 'invoice_id'])

    if date_col and amount_col and cust_id_col and invoice_col:
        df['Invoice Date'] = pd.to_datetime(df[date_col], errors='coerce')
        snapshot_date = df['Invoice Date'].max() + pd.Timedelta(days=1)

        rfm = df.groupby(cust_id_col).agg({
            'Invoice Date': lambda x: (snapshot_date - x.max()).days,
            invoice_col: 'nunique',
            amount_col: 'sum'
        })

        rfm.columns = ['Recency', 'Frequency', 'Monetary']
        rfm = rfm.reset_index()

        st.subheader("RFM Table")
        st.dataframe(rfm.head(10))

        # Scaling
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

        # KMeans
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

        st.subheader("Cluster Summary")
        st.dataframe(rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean())

        # Visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(
            rfm['Frequency'],
            rfm['Monetary'],
            c=rfm['Cluster'],
            cmap='viridis',
            s=100,
            alpha=0.6
        )
        ax.set_title("Customer Segmentation (RFM Analysis)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Frequency (Number of Purchases)")
        ax.set_ylabel("Monetary (Total Spent)")
        plt.colorbar(scatter, ax=ax, label='Cluster')
        st.pyplot(fig)
    else:
        st.error("⚠️ Required columns for RFM analysis not found")

# =================================================
# 6. DEMAND FORECASTING
# =================================================
elif menu == "Demand Forecasting":

    if not PROPHET_AVAILABLE:
        st.error("⚠️ Prophet is not installed. Please install it with: pip install prophet")
    else:
        st.header("📉 Demand Forecasting")

        date_col = get_date_column()
        amount_col = get_column_safe('Total_Amount', ['total_amount', 'Amount', 'amount'])

        if date_col and amount_col:
            try:
                df['Invoice Date'] = pd.to_datetime(df[date_col], errors='coerce')
                forecast_data = df.groupby('Invoice Date')[amount_col].sum().reset_index()
                forecast_data.columns = ['ds', 'y']
                forecast_data = forecast_data.sort_values('ds')

                st.subheader("Forecast Data")
                st.dataframe(forecast_data.head())

                # Prophet Model
                with st.spinner("🔄 Training Prophet model..."):
                    model = Prophet(yearly_seasonality=True, interval_width=0.95)
                    model.fit(forecast_data)

                    future = model.make_future_dataframe(periods=30)
                    forecast = model.predict(future)

                st.subheader("30-Day Forecast Results")
                st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(35))

                # Forecast Plot
                fig1 = model.plot(forecast)
                fig1.suptitle("Demand Forecast", fontsize=16, fontweight='bold')
                st.pyplot(fig1)

                # Components
                fig2 = model.plot_components(forecast)
                st.pyplot(fig2)

                st.success("✅ Forecasting Completed Successfully")

            except Exception as e:
                st.error(f"⚠️ Forecasting Error: {str(e)}")
        else:
            st.error("⚠️ Required date or amount columns not found")

# =================================================
# 7. CHURN PREDICTION
# =================================================
elif menu == "Churn Prediction":

    st.header("⚠️ Churn Prediction")

    amount_col = get_column_safe('Total_Amount', ['total_amount', 'Amount', 'amount'])
    cust_id_col = get_column_safe('Customer ID', ['customer_id', 'CustomerID', 'Customer_ID'])
    invoice_col = get_column_safe('Invoice', ['invoice', 'Invoice_ID', 'invoice_id'])
    churn_col = get_column_safe('Churn', ['churn', 'is_churn'])

    if amount_col and cust_id_col and invoice_col and churn_col:
        try:
            customer_data = df.groupby(cust_id_col).agg({
                amount_col: 'sum',
                invoice_col: 'nunique',
                churn_col: 'max'
            }).reset_index()

            customer_data.columns = [
                'Customer_ID',
                'Total_Spent',
                'Total_Orders',
                'Churn'
            ]

            X = customer_data[['Total_Spent', 'Total_Orders']]
            y = customer_data['Churn']

            # Check if we have both classes
            if len(y.unique()) < 2:
                st.warning("⚠️ Only one class present in Churn column. Cannot train classifier.")
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=0.2,
                    random_state=42
                )

                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)

                predictions = model.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)

                st.subheader("Model Performance")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Accuracy", f"{round(accuracy * 100, 2)}%")
                with col2:
                    st.metric("Test Samples", len(y_test))

                # Churn Distribution
                st.subheader("Churn Distribution")
                churn_counts = customer_data['Churn'].value_counts()

                fig, ax = plt.subplots()
                churn_counts.plot(kind='bar', ax=ax, color=['green', 'red'])
                ax.set_title("Customer Churn Distribution", fontsize=14, fontweight='bold')
                ax.set_xlabel("Churn Status (0=No, 1=Yes)")
                ax.set_ylabel("Count")
                plt.xticks(rotation=0)
                st.pyplot(fig)

                # Feature Importance
                st.subheader("Feature Importance")
                feature_importance = pd.DataFrame({
                    'Feature': ['Total_Spent', 'Total_Orders'],
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)

                fig, ax = plt.subplots()
                feature_importance.plot(x='Feature', y='Importance', kind='bar', ax=ax, legend=False)
                ax.set_title("Feature Importance", fontsize=14, fontweight='bold')
                ax.set_ylabel("Importance")
                plt.xticks(rotation=45)
                st.pyplot(fig)

                st.success("✅ Churn Prediction Completed")

        except Exception as e:
            st.error(f"⚠️ Churn Prediction Error: {str(e)}")
    else:
        st.error("⚠️ Required columns for Churn Prediction not found")
