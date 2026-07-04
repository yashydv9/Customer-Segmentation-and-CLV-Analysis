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

<img width="592" height="197" alt="image" src="https://github.com/user-attachments/assets/39cc4215-7247-4bbd-a851-91fc6c18450a" />

If your dataset uses different column names, you can map them directly inside the sidebar of the Streamlit app.

# How to Run the Project
## 1. Clone the Repository
```
git clone https://github.com/your-username/customer-segmentation-clv.git
```
 
```
cd customer-segmentation-clv
```

## 2. Create a Virtual Environment
```
python -m venv venv
```

### Activate the environment:
#### For Windows:
```
venv\Scripts\activate
```

#### For macOS or Linux:
```
source venv/bin/activate
```

## 3. Install Required Libraries
```
pip install streamlit pandas numpy matplotlib seaborn scikit-learn plotly
```

## 4. Run the Streamlit App
```
streamlit run Customer_Segmentation.py
```

After running the command, Streamlit will open the app in your browser.

## 5. Upload the Dataset
Upload a CSV file containing transaction data.

### Recommended columns:
customer_id
transaction_date
tran_amount

If your CSV has different names, update the column mapping fields in the sidebar.

## 6. Configure Analysis Settings
Inside the sidebar, select:
Number of clusters
Analysis period
Recency weight
Frequency weight
Monetary weight
Then click Run Full Analysis.

## 7. Review the Results
The app provides multiple result sections:
Cluster Summary: View customer count, average recency, frequency, and monetary value for each cluster
Visualizations: Explore pie charts, scatter plots, and 3D customer segment charts
Customer Details: Filter customers by cluster and sort by monetary value, frequency, recency, or RFM score
Recommendations: Get marketing actions for each segment

## 8. Download Outputs
You can download:
Full customer segmentation results
Cluster summary
Individual cluster customer data

# Methodology
## Data Preprocessing
The app loads transaction data, converts the transaction date column into datetime format, removes invalid date records, and filters the dataset based on the selected analysis period.

## RFM Calculation
For each customer, the app calculates:
**Recency**: Number of days since the last transaction
**Frequency**: Total number of transactions
**Monetary**: Total transaction amount

## Feature Scaling
The RFM values are log-transformed using log1p to reduce skewness. The values are then standardized using StandardScaler.

# K-Means Clustering
K-Means clustering is applied to the scaled RFM values. Each customer is assigned to a cluster, and cluster summaries are generated to interpret segment behavior.
Segment Interpretation
Clusters are interpreted based on their average recency, frequency, and monetary value. The app assigns business-friendly segment labels and recommends marketing actions for each group.

# Business Use Cases
This project can help businesses:
Identify high-value customers
Build targeted marketing campaigns
Improve customer retention
Detect inactive or low-value customers
Personalize offers and promotions
Prioritize customers for loyalty programs
Export segmented customer lists for campaign execution

# Future Improvements
Add CLV prediction using regression or probabilistic models
Add churn prediction
Add campaign response prediction
Support Excel file uploads
Add authentication for business users
Store previous analyses in a database
Deploy the app on Streamlit Community Cloud

# Conclusion
This project demonstrates how machine learning and customer analytics can be used to convert raw transaction data into actionable customer segments. By combining RFM analysis, K-Means clustering, interactive dashboards, and recommendation logic, the app provides a practical way to understand customer behavior and support data-driven marketing decisions.

Download individual cluster data
Generate actionable marketing recommendations for each customer segment
