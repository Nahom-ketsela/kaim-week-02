import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def basic_data_overview(df):
    print(df.info())
    print(df.describe())

def handle_missing_values(df):
    # Fill missing values
    df.fillna(df.mean(numeric_only=True), inplace=True)
    df.fillna(df.mode().iloc[0], inplace=True)  # For categorical columns

def treat_outliers(df, columns_to_check):
    for col in columns_to_check:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] > upper_bound, upper_bound, 
                           np.where(df[col] < lower_bound, lower_bound, df[col]))

def plot_correlation_matrix(df):
    correlation_matrix = df.corr()
    plt.figure(figsize=(16, 10))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.show()

def top_handsets(df):
    top_10_handsets = df['Handset Type'].value_counts().head(10)
    print("Top 10 Handsets:\n", top_10_handsets)

    top_3_manufacturers = df['Handset Manufacturer'].value_counts().head(3)
    print("Top 3 Handset Manufacturers:\n", top_3_manufacturers)

    for manufacturer in top_3_manufacturers.index:
        top_5_handsets = df[df['Handset Manufacturer'] == manufacturer]['Handset Type'].value_counts().head(5)
        print(f"Top 5 Handsets for {manufacturer}:\n", top_5_handsets)

def user_aggregation_metrics(df):
    user_metrics = df.groupby('MSISDN/Number').agg({
        'Bearer Id': 'count',  # Number of sessions
        'Dur. (ms)': 'sum',  # Total session duration
        'Total DL (Bytes)': 'sum',  # Total download data
        'Total UL (Bytes)': 'sum',  # Total upload data
    }).reset_index()

    user_metrics['Total Data Volume (Bytes)'] = user_metrics['Total DL (Bytes)'] + user_metrics['Total UL (Bytes)']
    print(user_metrics.head())
    return user_metrics

def decile_segmentation(user_metrics):
    user_metrics['Decile'] = pd.qcut(user_metrics['Dur. (ms)'], 10, labels=False)
    decile_data = user_metrics.groupby('Decile').agg({'Total Data Volume (Bytes)': 'sum'}).reset_index()
    print(decile_data)

def univariate_analysis(df):
    numeric_columns = df.select_dtypes(include=['float64']).columns
    for col in numeric_columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f"Distribution of {col}")
        plt.show()

def bivariate_analysis(df):
    apps = ['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)',
            'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']
    for app in apps:
        sns.scatterplot(data=df, x=app, y='Total DL (Bytes) + Total UL (Bytes)')
        plt.title(f"{app} vs Total Data Volume")
        plt.show()

def application_correlation_analysis(df):
    app_data = df[['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)',
                   'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']]
    correlation_matrix = app_data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix for Application Data")
    plt.show()

def pca_analysis(df):
    app_data = df[['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)',
                   'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']]

    # Standardize data
    pca_data = StandardScaler().fit_transform(app_data)

    # PCA transformation
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(pca_data)

    print("Explained Variance Ratio:", pca.explained_variance_ratio_)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1])
    plt.title("PCA Result")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()

    print("PCA Interpretation:")
    print("1. The first principal component explains {:.2f}% of the variance and relates to overall data usage patterns.".format(pca.explained_variance_ratio_[0] * 100))
    print("2. The second principal component explains {:.2f}% of the variance and focuses on specific app usage patterns.".format(pca.explained_variance_ratio_[1] * 100))
    print("3. Together, these two components capture {:.2f}% of the variance, providing a reduced dimensional view of user behavior.".format(sum(pca.explained_variance_ratio_) * 100))
