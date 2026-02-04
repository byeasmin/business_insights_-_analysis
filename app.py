import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from datetime import timedelta
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

# Page config
st.set_page_config(page_title="Market Basket Analysis", layout="wide", page_icon="📊")
st.title("📊 Business Insights Analysis Dashboard")
st.markdown("An interactive dashboard for Market Basket & Sales Analysis")

# Load and clean data
@st.cache_data
def load_data():
    df = pd.read_csv("OnlineRetail.csv", encoding='ISO-8859-1')
    df = df.dropna(subset=["CustomerID"])
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    df = df[df['Quantity'] > 0]
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Revenue'] = df['Quantity'] * df['UnitPrice']
    return df

df = load_data()

# Sidebar for navigation
st.sidebar.title("Navigation")
sections = ["Sales Performance", "Time-Series Analysis", "Customer Segmentation",
            "Basket Analysis", "Country-Level Analysis", "Price & Quantity Insights",
            "Fraud Detection", "Customer Retention"]
choice = st.sidebar.radio("Go to", sections)

# Section: Sales Performance
if choice == "Sales Performance":
    st.header("1️⃣ Sales Performance Analysis")
    
    # KPI cards
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Revenue", f"${df['Revenue'].sum():,.0f}")
    col2.metric("Total Orders", f"{df['InvoiceNo'].nunique()}")
    col3.metric("Total Customers", f"{df['CustomerID'].nunique()}")
    
    # Top products
    st.subheader("Top Products")
    top_products_qty = df.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(10)
    top_products_rev = df.groupby("Description")["Revenue"].sum().sort_values(ascending=False).head(10)
    
    col1, col2 = st.columns(2)
    fig_qty = px.bar(top_products_qty, x=top_products_qty.index, y=top_products_qty.values,
                     labels={"x":"Product","y":"Quantity"}, title="Top 10 Products by Quantity")
    fig_rev = px.bar(top_products_rev, x=top_products_rev.index, y=top_products_rev.values,
                     labels={"x":"Product","y":"Revenue"}, title="Top 10 Products by Revenue")
    col1.plotly_chart(fig_qty, use_container_width=True)
    col2.plotly_chart(fig_rev, use_container_width=True)
    
    # Top customers
    st.subheader("Top Customers")
    top_customers = df.groupby("CustomerID")["Revenue"].sum().sort_values(ascending=False).head(10)
    fig_cust = px.bar(top_customers, x=top_customers.index, y=top_customers.values,
                      labels={"x":"CustomerID","y":"Revenue"}, title="Top 10 Customers")
    st.plotly_chart(fig_cust, use_container_width=True)
    
    # Best-selling days & hours
    df['DayOfWeek'] = df['InvoiceDate'].dt.day_name()
    df['Hour'] = df['InvoiceDate'].dt.hour
    sales_by_day = df.groupby("DayOfWeek")["Revenue"].sum()
    sales_by_hour = df.groupby("Hour")["Revenue"].sum()
    
    col1, col2 = st.columns(2)
    fig_day = px.bar(sales_by_day, x=sales_by_day.index, y=sales_by_day.values, title="Sales by Day")
    fig_hour = px.bar(sales_by_hour, x=sales_by_hour.index, y=sales_by_hour.values, title="Sales by Hour")
    col1.plotly_chart(fig_day, use_container_width=True)
    col2.plotly_chart(fig_hour, use_container_width=True)
    
    # Product categories
    st.subheader("Most Profitable Product Categories")
    def get_category(desc):
        keywords = ["HOLDER","SET","CUP","BOTTLE","LANTERN","HEART"]
        for k in keywords:
            if k in str(desc).upper():
                return k
        return "OTHER"
    df['Category'] = df['Description'].apply(get_category)
    category_revenue = df.groupby("Category")["Revenue"].sum().sort_values(ascending=False)
    fig_cat = px.bar(category_revenue, x=category_revenue.index, y=category_revenue.values,
                     title="Revenue by Category")
    st.plotly_chart(fig_cat, use_container_width=True)

# Section: Time-Series Analysis
elif choice == "Time-Series Analysis":
    st.header("2️⃣ Time-Series & Seasonal Analysis")
    
    df['Month'] = df['InvoiceDate'].dt.to_period("M")
    monthly_sales = df.groupby("Month")["Revenue"].sum().to_timestamp()
    fig_month = px.line(monthly_sales, x=monthly_sales.index, y=monthly_sales.values, title="Monthly Sales Trend")
    st.plotly_chart(fig_month)
    
    yoy_growth = monthly_sales.pct_change(12).dropna()
    mom_growth = monthly_sales.pct_change().dropna()
    fig_yoy = px.line(yoy_growth, x=yoy_growth.index, y=yoy_growth.values, title="YoY Growth")
    fig_mom = px.line(mom_growth, x=mom_growth.index, y=mom_growth.values, title="MoM Growth")
    
    col1, col2 = st.columns(2)
    col1.plotly_chart(fig_yoy, use_container_width=True)
    col2.plotly_chart(fig_mom, use_container_width=True)

# --- Section: Customer Segmentation ---
elif choice == "Customer Segmentation":
    st.header("3️⃣ Customer Segmentation")
    snapshot_date = df['InvoiceDate'].max() + timedelta(days=1)
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'count',
        'Revenue': 'sum'
    })
    rfm.columns = ['Recency','Frequency','Monetary']
    diversity = df.groupby("CustomerID")["Description"].nunique()
    rfm['Diversity'] = diversity
    st.dataframe(rfm.head(10))

# Section: Basket Analysis 
elif choice == "Basket Analysis":
    st.header("4️⃣ Basket Analysis")
    top_products = df.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(500).index
    df_small = df[df['Description'].isin(top_products)]
    basket = df_small.groupby(['InvoiceNo','Description'])['Quantity'].sum().unstack().fillna(0)
    basket = (basket > 0)
    try:
        frequent_itemsets = apriori(basket, min_support=0.02, use_colnames=True)
    except MemoryError:
        frequent_itemsets = fpgrowth(basket, min_support=0.02, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    st.subheader("Top 10 Rules by Lift")
    st.dataframe(rules.sort_values("lift", ascending=False).head(10))

# Section: Country-Level Analysis 
elif choice == "Country-Level Analysis":
    st.header("5️⃣ Country-Level Analysis")
    country_sales = df.groupby("Country")["Revenue"].sum().sort_values(ascending=False)
    avg_transaction_size = df.groupby("Country")["Revenue"].mean()
    
    col1, col2 = st.columns(2)
    fig_country_sales = px.bar(country_sales, x=country_sales.index, y=country_sales.values, title="Revenue by Country")
    fig_avg_trans = px.bar(avg_transaction_size, x=avg_transaction_size.index, y=avg_transaction_size.values,
                           title="Avg Transaction Size by Country")
    col1.plotly_chart(fig_country_sales, use_container_width=True)
    col2.plotly_chart(fig_avg_trans, use_container_width=True)

# Section: Price & Quantity Insights
elif choice == "Price & Quantity Insights":
    st.header("6️⃣ Price & Quantity Insights")
    fig_price = px.scatter(df, x="UnitPrice", y="Quantity", log_x=True, log_y=True, title="Price vs Quantity")
    st.plotly_chart(fig_price)
    st.subheader("Outliers")
    bulk_orders = df[(df["Quantity"] > df["Quantity"].quantile(0.99)) | (df["Revenue"] > df["Revenue"].quantile(0.99))]
    st.dataframe(bulk_orders.head(10))

# Section: Fraud Detection 
elif choice == "Fraud Detection":
    st.header("7️⃣ Fraud / Anomaly Detection")
    fraud = df[(df["Quantity"] <= 0) | (df["UnitPrice"] <= 0)]
    st.dataframe(fraud.head(10))

# Section: Customer Retention
elif choice == "Customer Retention":
    st.header("8️⃣ Customer Retention & Churn")
    repeat_customers = df.groupby("CustomerID")["InvoiceNo"].nunique()
    repeat_rate = (repeat_customers[repeat_customers > 1].count() / repeat_customers.count()) * 100
    st.metric("Repeat Purchase Rate", f"{repeat_rate:.2f}%")
    
    df_sorted = df.sort_values(["CustomerID","InvoiceDate"])
    df_sorted['PrevInvoice'] = df_sorted.groupby('CustomerID')['InvoiceDate'].shift(1)
    df_sorted['DaysGap'] = (df_sorted['InvoiceDate'] - df_sorted['PrevInvoice']).dt.days
    avg_gap = df_sorted.groupby('CustomerID')['DaysGap'].mean().dropna()
    st.write("Average time gap between purchases for loyal customers (days):")
    st.dataframe(avg_gap.head(10))

st.success("✅ Dashboard Ready!.")
