# Customer-Segmentation-and-CLV-Analysis
This project is a customer analytics application that automates customer segmentation using RFM analysis and K Means clustering. Users can upload transaction data through a Stream lit dashboard to generate customer segments, while an exploratory notebook explains the complete analysis workflow and insights.

### App link - https://customer-segmentation-and-clv-analysis.streamlit.app/

# Project Overview
Customer segmentation is an important part of marketing analytics. Instead of treating every customer the same way, this project groups customers based on their purchasing behavior.
The analysis uses three key RFM metrics:
Recency: How recently a customer made a purchase
Frequency: How often a customer makes purchases
Monetary: How much a customer has spent
These metrics are then used with K-Means clustering to divide customers into meaningful groups such as high-value loyal customers, frequent buyers, recent low-spend customers, average customers, and low-value or inactive customers.

# Features
Upload transaction CSV files through a Streamlit interface
Map custom column names for customer ID, transaction date, and transaction amount
Select analysis period such as last 90, 180, 365 days, or all time
Adjust RFM weights for recency, frequency, and monetary value
Choose the number of clusters for K-Means segmentation
View customer segment summaries and cluster-level statistics
Explore interactive visualizations using Plotly
Analyze customers using 2D and 3D cluster charts
View detailed customer data by cluster

# Dataset Format
The Streamlit app expects a transaction dataset in CSV format. The required columns are:

## Column	             ## Description
customer_id	           Unique customer identifier
transaction_date	     Date of transaction
tran_amount	           Transaction amount
Download full segmentation results as CSV
Download individual cluster data
Generate actionable marketing recommendations for each customer segment
