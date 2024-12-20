import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

#Aggregate Engagement Metrics Per Customer
def aggregate_engagement_metrics(df):
    user_metrics = df.groupby('MSISDN/Number').agg({
        'Bearer Id': 'count',  # Sessions frequency
        'Dur. (ms)': 'sum',  # Total session duration
        'Total DL (Bytes)': 'sum',  # Total download data
        'Total UL (Bytes)': 'sum',  # Total upload data
    }).reset_index()

    user_metrics['Total Traffic (Bytes)'] = user_metrics['Total DL (Bytes)'] + user_metrics['Total UL (Bytes)']
    return user_metrics

# Report Top 10 Customers
def report_top_users(user_metrics):
    print("Top 10 Users by Sessions Frequency:")
    print(user_metrics.nlargest(10, 'Bearer Id'))

    print("\nTop 10 Users by Session Duration:")
    print(user_metrics.nlargest(10, 'Dur. (ms)'))

    print("\nTop 10 Users by Total Traffic:")
    print(user_metrics.nlargest(10, 'Total Traffic (Bytes)'))

# Normalize Metrics and Apply K-Means
def normalize_metrics(user_metrics):
    # Define the columns to be normalized
    columns_to_normalize = ['Bearer Id', 'Dur. (ms)', 'Total Traffic (Bytes)']
    
    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()
    
    # Fit and transform the data
    normalized_metrics = scaler.fit_transform(user_metrics[columns_to_normalize])
    
    # Create a DataFrame with the normalized data
    normalized_df = pd.DataFrame(normalized_metrics, columns=columns_to_normalize)
    
    return normalized_df

def apply_kmeans(normalized_metrics, k=3):
    kmeans = KMeans(n_clusters=k, random_state=42)
    return kmeans.fit_predict(normalized_metrics)

# Cluster Analysis
def analyze_clusters(user_metrics):
    cluster_summary = user_metrics.groupby('Cluster').agg({
        'Bearer Id': ['min', 'max', 'mean', 'sum'],
        'Dur. (ms)': ['min', 'max', 'mean', 'sum'],
        'Total Traffic (Bytes)': ['min', 'max', 'mean', 'sum']
    })
    print(cluster_summary)

    plt.figure(figsize=(10, 6))
    sns.countplot(data=user_metrics, x='Cluster')
    plt.title("Distribution of Users Across Clusters")
    plt.xlabel("Cluster")
    plt.ylabel("Number of Users")
    plt.show()

# Application-Based Engagement
def top_application_users(df):
    apps = ['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)',
            'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']

    app_engagement = {}
    for app in apps:
        app_engagement[app] = df.groupby('MSISDN/Number')[app].sum().nlargest(10)
        print(f"Top 10 Users for {app}:\n{app_engagement[app]}")

    return app_engagement

def plot_top_apps(df):
    apps = ['Social Media DL (Bytes)', 'Youtube DL (Bytes)', 'Netflix DL (Bytes)']
    total_app_traffic = df[apps].sum()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=total_app_traffic.index, y=total_app_traffic.values, palette="viridis")
    plt.title("Top 3 Most Used Applications")
    plt.xlabel("Application")
    plt.ylabel("Total Traffic (Bytes)")
    plt.show()

# Elbow Method for Optimized K
def elbow_method(normalized_metrics):
    distortions = []
    K = range(1, 10)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(normalized_metrics)
        distortions.append(kmeans.inertia_)

    # Plot the elbow curve
    plt.figure(figsize=(8, 5))
    plt.plot(K, distortions, marker='o')
    plt.title("Elbow Method for Optimal K")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Distortion")
    plt.show()
