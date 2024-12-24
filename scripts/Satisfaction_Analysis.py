import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean

# Aggregate Engagement Metrics
def aggregate_engagement_metrics(df):
    user_metrics = df.groupby('MSISDN/Number').agg({
        'Bearer Id': 'count',  # Sessions frequency
        'Dur. (ms)': 'sum',  # Total session duration
        'Total DL (Bytes)': 'sum',  # Total download data
        'Total UL (Bytes)': 'sum'   # Total upload data
    }).reset_index()
    user_metrics['Total Traffic (Bytes)'] = user_metrics['Total DL (Bytes)'] + user_metrics['Total UL (Bytes)']
    return user_metrics

# Aggregate Experience Metrics
def aggregate_experience_metrics(df):
    experience_metrics = df.groupby('MSISDN/Number').agg({
        'Avg RTT DL (ms)': 'mean',  
        'Avg RTT UL (ms)': 'mean',   
        'Avg Bearer TP DL (kbps)': 'mean',   
        'Avg Bearer TP UL (kbps)': 'mean',   
        'TCP DL Retrans. Vol (Bytes)': 'mean',   
        'TCP UL Retrans. Vol (Bytes)': 'mean'    
    }).reset_index()
    return experience_metrics

# Normalize Metrics
def normalize_metrics(df, columns_to_normalize):
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(df[columns_to_normalize])
    normalized_df = pd.DataFrame(normalized_data, columns=columns_to_normalize)
    return normalized_df, scaler

# Apply KMeans Clustering
def apply_kmeans(normalized_metrics, k=3):
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(normalized_metrics)
    return cluster_labels, kmeans

# Calculate Engagement Scores
def calculate_engagement_scores(normalized_df, cluster_labels, kmeans):
    # Identify the less engaged cluster
    cluster_means = normalized_df.groupby(cluster_labels).mean()
    less_engaged_cluster = cluster_means.mean(axis=1).idxmin()

    # Centroid of the less engaged cluster
    less_engaged_centroid = kmeans.cluster_centers_[less_engaged_cluster]

    # Calculate Euclidean distance to the less engaged cluster centroid
    distances = normalized_df.apply(lambda row: euclidean(row, less_engaged_centroid), axis=1)
    return distances

# Combine Engagement and Experience Metrics
def combine_metrics(engagement_metrics, experience_metrics):
    combined_metrics = pd.merge(engagement_metrics, experience_metrics, on='MSISDN/Number')
    return combined_metrics

# Full Pipeline Execution
def full_analysis_pipeline(data, k=3):
    # Aggregate metrics
    engagement_metrics = aggregate_engagement_metrics(data)
    experience_metrics = aggregate_experience_metrics(data)
    
    # Combine metrics
    combined_metrics = combine_metrics(engagement_metrics, experience_metrics)
    
    # Normalize metrics
    columns_to_normalize = ['Bearer Id', 'Dur. (ms)', 'Total Traffic (Bytes)', 
                            'Avg RTT DL (ms)', 'Avg RTT UL (ms)', 'Avg Bearer TP DL (kbps)', 
                            'Avg Bearer TP UL (kbps)', 'TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)']
    normalized_df, scaler = normalize_metrics(combined_metrics, columns_to_normalize)
    
    # Apply clustering
    cluster_labels, kmeans = apply_kmeans(normalized_df, k=k)
    
    # Calculate engagement scores
    engagement_scores = calculate_engagement_scores(normalized_df, cluster_labels, kmeans)
    
    # Add results to combined_metrics
    combined_metrics['Cluster'] = cluster_labels
    combined_metrics['Engagement Score'] = engagement_scores
    
    return combined_metrics
