import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

st.title("Customer Segmentation Dashboard")

st.markdown("Analyze customer groups based on income and spending behavior.")

st.sidebar.header("Dashboard Menu")
st.sidebar.write("Customer analytics using Machine Learning")

df = pd.read_csv("mall_customers.csv")

col1, col2 = st.columns(2)

with col1:
    st.metric("Total Customers", len(df))

with col2:
    st.metric("Clusters Created", 5)

st.subheader("Dataset Preview")

st.dataframe(df.head())

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

kmeans = KMeans(n_clusters=5, random_state=42)

df['Cluster'] = kmeans.fit_predict(X)

st.subheader("Customer Segmentation Graph")

fig, ax = plt.subplots(figsize=(10,7))

scatter = ax.scatter(
    df['Annual Income (k$)'],
    df['Spending Score (1-100)'],
    c=df['Cluster'],
    s=100
)

ax.set_xlabel("Annual Income")
ax.set_ylabel("Spending Score")
ax.set_title("Customer Groups")

st.pyplot(fig)

st.subheader("Clustered Customer Data")

st.dataframe(df)

st.subheader("Business Insights")

st.success("High income customers with high spending are premium buyers.")

st.info("Customers with high income but low spending may need targeted marketing.")

st.warning("Low spending customers can be engaged using discounts and offers.")