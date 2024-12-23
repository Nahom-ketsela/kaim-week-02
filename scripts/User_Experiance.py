import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


def aggregate_per_customer(df):
    """
    Aggregate metrics per customer, replacing missing values and handling outliers.

    Parameters:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Aggregated DataFrame.
    """
    
    # Perform the aggregation
    agg_df = df.groupby('MSISDN/Number').agg({
        'TCP DL Retrans. Vol (Bytes)': 'mean',
        'TCP UL Retrans. Vol (Bytes)': 'mean',
        'Avg RTT DL (ms)': 'mean',
        'Avg RTT UL (ms)': 'mean',
        'Handset Type': lambda x: x.mode()[0],
        'Avg Bearer TP DL (kbps)': 'mean',
        'Avg Bearer TP UL (kbps)': 'mean'
    }).reset_index()

    # Rename columns to more user-friendly names
    agg_df.rename(columns={
        'TCP DL Retrans. Vol (Bytes)': 'Average TCP Retransmission DL',
        'TCP UL Retrans. Vol (Bytes)': 'Average TCP Retransmission UL',
        'Avg RTT DL (ms)': 'Average RTT DL',
        'Avg RTT UL (ms)': 'Average RTT UL',
        'Avg Bearer TP DL (kbps)': 'Average Throughput DL',
        'Avg Bearer TP UL (kbps)': 'Average Throughput UL'
    }, inplace=True)

    return agg_df

def compute_top_bottom_frequent(df, column, n=10):
    """
    Compute top, bottom, and most frequent values in a column.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name to analyze.
        n (int): Number of values to compute.

    Returns:
        dict: Dictionary with top, bottom, and frequent values.
    """
    top_values = df[column].nlargest(n)
    bottom_values = df[column].nsmallest(n)
    frequent_values = df[column].value_counts().head(n)

    return {
        'Top': top_values,
        'Bottom': bottom_values,
        'Frequent': frequent_values
    }


def distribution_per_handset(df, metric_column, handset_column):
    """
    Compute the distribution of a metric per handset type.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        metric_column (str): Column name for the metric.
        handset_column (str): Column name for the handset type.

    Returns:
        pd.DataFrame: Aggregated distribution DataFrame.
    """
    distribution = df.groupby(handset_column)[metric_column].mean().reset_index()

    plt.figure(figsize=(12, 6))
    sns.barplot(x=handset_column, y=metric_column, data=distribution, palette="viridis")
    plt.title(f"Distribution of {metric_column} per Handset Type", fontsize=16)
    plt.xlabel("Handset Type", fontsize=12)
    plt.ylabel(f"Average {metric_column}", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    return distribution


def kmeans_clustering_experience(df, k=3):
    """
    Perform K-means clustering on user experience metrics and visualize the clusters.

    Parameters:
        df (pd.DataFrame): DataFrame containing experience metrics.
        k (int): Number of clusters for K-means.

    Returns:
        pd.DataFrame: DataFrame with cluster assignments.
        KMeans: Fitted KMeans model.
    """
    features = ['Average Throughput', 'Average TCP Retransmission', 'Average RTT']

    # Handle missing values
    df[features] = df[features].fillna(df[features].mean())

    # Standardize the features
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(df[features])

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    df['Cluster'] = kmeans.fit_predict(standardized_data)

    # Plot clusters
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df['Average Throughput'], 
                    y=df['Average TCP Retransmission'], 
                    hue=df['Cluster'], 
                    palette='viridis', 
                    s=100, 
                    alpha=0.8)
    plt.title('K-means Clustering of Users (K=3)', fontsize=16)
    plt.xlabel('Average Throughput', fontsize=12)
    plt.ylabel('Average TCP Retransmission', fontsize=12)
    plt.legend(title='Cluster', fontsize=10)
    plt.tight_layout()
    plt.show()

    return df, kmeans

def interpret_clusters(df):
    """
    Provide an interpretation of each cluster based on aggregated metrics.

    Parameters:
        df (pd.DataFrame): DataFrame containing clustered data.

    Returns:
        pd.DataFrame: Cluster interpretation.
    """
    cluster_summary = df.groupby('Cluster').agg({
        'Average Throughput': 'mean',
        'Average TCP Retransmission': 'mean',
        'Average RTT': 'mean'
    }).reset_index()

    print("Cluster Summary:\n", cluster_summary)
    return cluster_summary