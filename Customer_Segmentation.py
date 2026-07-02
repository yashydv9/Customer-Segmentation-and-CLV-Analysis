# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation App",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("📊 Automated Customer Segmentation Analyzer")
st.markdown("""
Upload your transaction CSV file to automatically perform RFM analysis and customer clustering.
The app will identify customer segments and provide actionable insights.
""")

# Sidebar for file upload and parameters
with st.sidebar:
    st.header("📁 Data Upload")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    st.header("⚙️ Analysis Parameters")
    
    # Column mapping options
    st.subheader("Column Mapping")
    st.markdown("If your CSV has different column names, map them below:")
    
    col1, col2 = st.columns(2)
    with col1:
        customer_col = st.text_input("Customer ID Column", value="customer_id")
        date_col = st.text_input("Date Column", value="transaction_date")
    with col2:
        amount_col = st.text_input("Amount Column", value="tran_amount")
        quantity_col = st.text_input("Quantity Column (optional)", value="")
    
    # Clustering parameters
    st.subheader("Clustering Settings")
    n_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=4)
    analysis_period = st.selectbox(
        "Analysis Period",
        ["Last 90 days", "Last 180 days", "Last 365 days", "All time"],
        index=2
    )
    
    # RFM parameters
    st.subheader("RFM Settings")
    recency_weight = st.slider("Recency Weight", 0.0, 2.0, 1.0, 0.1)
    frequency_weight = st.slider("Frequency Weight", 0.0, 2.0, 1.0, 0.1)
    monetary_weight = st.slider("Monetary Weight", 0.0, 2.0, 1.0, 0.1)
    
    if st.button("🚀 Run Full Analysis", type="primary"):
        st.session_state.run_analysis = True
    else:
        st.session_state.run_analysis = False

# Main app logic
if uploaded_file is not None:
    try:
        # Load the data
        df = pd.read_csv(uploaded_file)
        
        # Show data preview
        with st.expander("📋 Data Preview", expanded=True):
            st.dataframe(df.head(), use_container_width=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Unique Customers", df[customer_col].nunique())
            with col3:
                st.metric("Date Range", 
                         f"{df[date_col].min()} to {df[date_col].max()}")
        
        # Data preprocessing
        with st.spinner("Preprocessing data..."):
            # Convert date column
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            
            # Handle missing dates
            if df[date_col].isnull().any():
                st.warning(f"Found {df[date_col].isnull().sum()} records with invalid dates. These will be excluded.")
                df = df.dropna(subset=[date_col])
            
            # Filter by analysis period
            if analysis_period != "All time":
                days = int(analysis_period.split()[1])
                cutoff_date = df[date_col].max() - timedelta(days=days)
                df = df[df[date_col] >= cutoff_date]
            
            # Calculate RFM
            analysis_date = df[date_col].max() + timedelta(days=1)
            
            rfm = df.groupby(customer_col).agg({
                date_col: lambda x: (analysis_date - x.max()).days,
                customer_col: 'count',
                amount_col: 'sum'
            }).rename(columns={
                date_col: 'recency',
                customer_col: 'frequency',
                amount_col: 'monetary'
            }).reset_index()
            
            # Apply weights
            rfm['weighted_r'] = (rfm['recency'].rank(pct=True) * recency_weight)
            rfm['weighted_f'] = (rfm['frequency'].rank(pct=True) * frequency_weight)
            rfm['weighted_m'] = (rfm['monetary'].rank(pct=True) * monetary_weight)
            
            # Calculate composite score
            rfm['rfm_score'] = rfm['weighted_r'] + rfm['weighted_f'] + rfm['weighted_m']
        
        # Clustering analysis
        with st.spinner("Performing clustering analysis..."):
            # Prepare data for clustering
            clustering_data = rfm[['recency', 'frequency', 'monetary']].copy()
            
            # Log transformation
            for col in ['recency', 'frequency', 'monetary']:
                clustering_data[col] = np.log1p(clustering_data[col])
            
            # Standardize
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(clustering_data)
            
            # Perform K-Means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
            rfm['cluster'] = kmeans.fit_predict(scaled_data)
            
            # Calculate cluster centers in original scale
            cluster_centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
            
            # Create cluster summary
            cluster_summary = rfm.groupby('cluster').agg({
                customer_col: 'count',
                'recency': ['mean', 'std'],
                'frequency': ['mean', 'std'],
                'monetary': ['mean', 'std'],
                'rfm_score': 'mean'
            }).round(2)
            
            # Flatten column names
            cluster_summary.columns = ['_'.join(col).strip() for col in cluster_summary.columns.values]
            cluster_summary = cluster_summary.rename(columns={f'{customer_col}_count': 'customer_count'})
            
            # Add cluster interpretation
            def interpret_cluster(row):
                if row['recency_mean'] < rfm['recency'].quantile(0.25):
                    if row['monetary_mean'] > rfm['monetary'].quantile(0.75):
                        return "High Value Loyal"
                    else:
                        return "Recent but Low Spend"
                elif row['frequency_mean'] > rfm['frequency'].quantile(0.75):
                    return "Frequent Buyers"
                elif row['monetary_mean'] < rfm['monetary'].quantile(0.25):
                    return "Low Value"
                else:
                    return "Average Customers"
            
            cluster_summary['segment_type'] = cluster_summary.apply(interpret_cluster, axis=1)
        
        # Display results
        st.success("✅ Analysis completed successfully!")
        
        # Results in tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Cluster Summary", "📊 Visualizations", "👥 Customer Details", "💡 Recommendations"])
        
        with tab1:
            st.subheader("Cluster Analysis Summary")
            
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Clusters", n_clusters)
            with col2:
                st.metric("Total Customers", len(rfm))
            with col3:
                st.metric("Avg Recency (days)", f"{rfm['recency'].mean():.0f}")
            with col4:
                st.metric("Avg Monetary Value", f"${rfm['monetary'].mean():.2f}")
            
            # Cluster summary table
            st.dataframe(cluster_summary, use_container_width=True)
            
            # Cluster characteristics
            st.subheader("Cluster Characteristics")
            for cluster_id in sorted(rfm['cluster'].unique()):
                cluster_data = rfm[rfm['cluster'] == cluster_id]
                segment = cluster_summary.loc[cluster_id, 'segment_type']
                
                with st.expander(f"Cluster {cluster_id} - {segment} ({len(cluster_data)} customers)"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Avg Recency", f"{cluster_data['recency'].mean():.0f} days")
                    with col2:
                        st.metric("Avg Frequency", f"{cluster_data['frequency'].mean():.1f}")
                    with col3:
                        st.metric("Avg Monetary", f"${cluster_data['monetary'].mean():.2f}")
                    
                    # Top customers in this cluster
                    top_customers = cluster_data.nlargest(5, 'monetary')[['customer_id', 'recency', 'frequency', 'monetary']]
                    st.write(f"Top 5 customers by value:")
                    st.dataframe(top_customers, use_container_width=True)
        
        with tab2:
            st.subheader("Cluster Visualizations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Cluster distribution pie chart
                fig1 = px.pie(
                    cluster_summary.reset_index(),
                    values='customer_count',
                    names='segment_type',
                    title='Customer Distribution by Segment',
                    hole=0.3
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # RFM scatter plot
                fig2 = px.scatter(
                    rfm,
                    x='recency',
                    y='monetary',
                    color='cluster',
                    size='frequency',
                    hover_data=['customer_id', 'frequency'],
                    title='Recency vs Monetary Value by Cluster',
                    labels={'recency': 'Days Since Last Purchase', 'monetary': 'Total Spend'}
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            # 3D Scatter plot
            st.subheader("3D Cluster Visualization")
            fig3 = px.scatter_3d(
                rfm,
                x='recency',
                y='frequency',
                z='monetary',
                color='cluster',
                size='frequency',
                hover_data=['customer_id'],
                title='3D View of Customer Segments',
                labels={
                    'recency': 'Recency (days)',
                    'frequency': 'Frequency',
                    'monetary': 'Monetary Value ($)'
                }
            )
            st.plotly_chart(fig3, use_container_width=True)
        
        with tab3:
            st.subheader("Detailed Customer Data")
            
            # Filter options
            col1, col2 = st.columns(2)
            with col1:
                selected_cluster = st.selectbox(
                    "Filter by Cluster",
                    options=sorted(rfm['cluster'].unique())
                )
            with col2:
                sort_by = st.selectbox(
                    "Sort by",
                    ['monetary', 'frequency', 'recency', 'rfm_score']
                )
            
            # Display filtered data
            filtered_data = rfm[rfm['cluster'] == selected_cluster].sort_values(sort_by, ascending=False)
            st.dataframe(filtered_data, use_container_width=True)
            
            # Download option
            csv = filtered_data.to_csv(index=False)
            st.download_button(
                label="📥 Download This Cluster's Data",
                data=csv,
                file_name=f"cluster_{selected_cluster}_customers.csv",
                mime="text/csv"
            )
        
        with tab4:
            st.subheader("Actionable Recommendations")
            
            # Generate recommendations based on clusters
            recommendations = []
            
            for cluster_id in sorted(rfm['cluster'].unique()):
                cluster_data = rfm[rfm['cluster'] == cluster_id]
                segment = cluster_summary.loc[cluster_id, 'segment_type']
                
                if segment == "High Value Loyal":
                    rec = {
                        "segment": segment,
                        "action": "VIP Treatment",
                        "suggestions": [
                            "Offer exclusive early access to new products",
                            "Create a VIP loyalty program with tiered rewards",
                            "Assign dedicated account managers",
                            "Request testimonials and referrals"
                        ],
                        "budget": "High",
                        "expected_roi": "Very High"
                    }
                elif "Recent" in segment:
                    rec = {
                        "segment": segment,
                        "action": "Engagement Boost",
                        "suggestions": [
                            "Send welcome series emails",
                            "Offer first-time buyer discount on next purchase",
                            "Introduce loyalty program",
                            "Request product reviews"
                        ],
                        "budget": "Medium",
                        "expected_roi": "High"
                    }
                elif "Frequent" in segment:
                    rec = {
                        "segment": segment,
                        "action": "Retention Focus",
                        "suggestions": [
                            "Offer bundle discounts",
                            "Create a subscription option",
                            "Send personalized reorder reminders",
                            "Provide volume-based discounts"
                        ],
                        "budget": "Medium",
                        "expected_roi": "High"
                    }
                else:
                    rec = {
                        "segment": segment,
                        "action": "Reactivation",
                        "suggestions": [
                            "Send win-back emails with special offers",
                            "Conduct customer feedback surveys",
                            "Offer limited-time discounts",
                            "Re-engage through social media"
                        ],
                        "budget": "Low",
                        "expected_roi": "Medium"
                    }
                
                rec["customer_count"] = len(cluster_data)
                rec["avg_value"] = f"${cluster_data['monetary'].mean():.2f}"
                recommendations.append(rec)
            
            # Display recommendations
            for rec in recommendations:
                with st.expander(f"{rec['segment']} ({rec['customer_count']} customers)"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Action Priority", rec['action'])
                    with col2:
                        st.metric("Budget Level", rec['budget'])
                    with col3:
                        st.metric("Expected ROI", rec['expected_roi'])
                    
                    st.markdown("**Recommended Actions:**")
                    for suggestion in rec['suggestions']:
                        st.markdown(f"- {suggestion}")
        
        # Download full results
        st.subheader("📥 Download Results")
        col1, col2 = st.columns(2)
        
        with col1:
            full_csv = rfm.to_csv(index=False)
            st.download_button(
                label="Download Full Analysis (CSV)",
                data=full_csv,
                file_name="customer_segmentation_results.csv",
                mime="text/csv"
            )
        
        with col2:
            summary_csv = cluster_summary.to_csv()
            st.download_button(
                label="Download Cluster Summary (CSV)",
                data=summary_csv,
                file_name="cluster_summary.csv",
                mime="text/csv"
            )
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.info("""
        Common issues:
        1. Column names don't match - check your column mapping
        2. Date format issues - ensure dates are in YYYY-MM-DD format
        3. Missing values - check for empty cells in required columns
        """)
else:
    # Show sample data structure when no file is uploaded
    st.info("👈 Please upload a CSV file to begin analysis.")
    
    with st.expander("📋 Expected CSV Format"):
        st.markdown("""
        Your CSV should have at least these columns (names can be customized):
        
        | customer_id | transaction_date | tran_amount |
        |------------|------------------|-------------|
        | CUST_001   | 2023-01-15       | 49.99       |
        | CUST_002   | 2023-01-16       | 129.99      |
        | CUST_001   | 2023-02-01       | 24.99       |
        
        **Optional columns:**
        - `quantity`: Number of items purchased
        - `product_id`: Product identifier
        - `category`: Product category
        """)
        
        # Create sample data
        sample_data = pd.DataFrame({
            'customer_id': ['CUST_001', 'CUST_002', 'CUST_001', 'CUST_003', 'CUST_002'],
            'transaction_date': ['2023-01-15', '2023-01-16', '2023-02-01', '2023-02-15', '2023-03-01'],
            'tran_amount': [49.99, 129.99, 24.99, 79.99, 199.99]
        })
        st.dataframe(sample_data, use_container_width=True)