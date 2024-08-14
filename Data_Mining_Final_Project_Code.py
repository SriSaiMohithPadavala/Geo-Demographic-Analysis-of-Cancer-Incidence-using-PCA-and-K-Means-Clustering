#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 20:35:44 2024

@author: andeysaikiran
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dask.dataframe import DataFrame as dd
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns


# Load the Chronic Disease Indicators (CDI) data from a CSV file into a DataFrame
cdi_df = pd.read_csv(r'/Users/andeysaikiran/Downloads/U.S._Chronic_Disease_Indicators__CDI___2023_dropped.csv')
# Display information about the CDI DataFrame to understand its structure and types
cdi_df.info()

# Load population data from a CSV file into a separate DataFrame
population = pd.read_csv(r'/Users/andeysaikiran/Downloads/population.csv')
# Display information about the population DataFrame to understand its structure and types
population.info()

# Drop unnecessary columns 'Unnamed: 13' and 'Unnamed: 14' from the CDI DataFrame
# This step is necessary to clean the data, removing columns that are not needed and might have been added by mistake
cdi_df = cdi_df.drop(['Unnamed: 13', 'Unnamed: 14'], axis=1)

# Display the first few rows of the CDI DataFrame to verify the structure and data after dropping columns
cdi_df.head()

# Calculate the average values for 'LowConfidenceLimit' and 'HighConfidenceLimit' for each 'Topic'
# This step is essential for understanding the range of confidence across different topics
average_values = cdi_df.groupby('Topic')[['LowConfidenceLimit', 'HighConfidenceLimit']].mean()

# Fill missing values in 'LowConfidenceLimit' with the average values calculated for each 'Topic'
# This step helps in handling missing data by providing a reasonable estimate based on the available data
cdi_df['LowConfidenceLimit'] = cdi_df['LowConfidenceLimit'].fillna(cdi_df['Topic'].map(average_values['LowConfidenceLimit']))

# Similarly, fill missing values in 'HighConfidenceLimit' with the average values for each 'Topic'
# This ensures that the dataset is complete and can be used for further analysis without missing data issues
cdi_df['HighConfidenceLimit'] = cdi_df['HighConfidenceLimit'].fillna(cdi_df['Topic'].map(average_values['HighConfidenceLimit']))

# Display information about the CDI DataFrame after filling missing values
# This is to verify the changes made and ensure the DataFrame is now complete and ready for analysis
cdi_df.info()


# --------- Data Preparation and Cleaning ---------
# Load unique location descriptions from both datasets to identify common and distinct locations
pop_list = population['LocationDesc'].unique()
cdi_list = cdi_df['LocationDesc'].unique()

# Convert lists to sets to facilitate set operations
set_1 = set(pop_list)
set_2 = set(cdi_list)

# Identify common and distinct locations between the two datasets
common = set_1.intersection(set_2)
not_common = set_1.symmetric_difference(set_2)

# Output common and distinct locations for verification
print("Common Locations:", common)
print("Not Common Locations:", not_common)

# Strip whitespace from string columns in both dataframes to ensure consistency
for col in cdi_df.columns:
    if cdi_df[col].dtype == "object":
        cdi_df[col] = cdi_df[col].str.strip()

for col in population.columns:
    if population[col].dtype == "object":
        population[col] = population[col].str.strip()

# --------- Data Merging ---------
# Merge the chronic disease data with the population data on the 'LocationDesc' column
cdi_merged_df = pd.merge(cdi_df, population, on='LocationDesc', how='inner')

# --------- Data Cleaning and Conversion ---------
# Fill missing 'DataValue' entries with 0
cdi_merged_df['DataValue'] = cdi_merged_df['DataValue'].fillna(0)

# Convert 'DataValue' to numeric, setting errors to NaN to handle non-numeric entries gracefully
cdi_merged_df['DataValue'] = pd.to_numeric(cdi_merged_df['DataValue'], errors='coerce')

# Rename the population column for clarity
cdi_merged_df.rename(columns={'Unnamed: 1': 'Population'}, inplace=True)

# --------- Incidence Rate Calculation ---------
# Calculate disease counts grouped by location, year, and topic
disease_counts = cdi_merged_df.groupby(['LocationDesc', 'YearStart', 'Topic']).size().reset_index(name='Count')

# Merge disease counts with population data to calculate incidence rates
disease_counts = pd.merge(disease_counts, cdi_merged_df[['LocationDesc', 'YearStart', 'Population']], on=['LocationDesc', 'YearStart'], how='inner')

# Calculate incidence rate per 100,000 population
disease_counts['IncidenceRate'] = (disease_counts['Count'] / disease_counts['Population']) * 100000

# Remove duplicate rows, if any
disease_counts = disease_counts.drop_duplicates()

# --------- Conversion to Dask DataFrames for Efficient Processing ---------
# Convert the merged DataFrame and disease counts DataFrame to Dask DataFrames for efficient parallel processing

# Assuming 'cdi_merged_df' and 'disease_counts' are your initial pandas DataFrames

# Filter 'disease_counts' DataFrame to include only the columns necessary for the merge
filtered_disease_counts = disease_counts[['LocationDesc', 'YearStart', 'Topic', 'IncidenceRate']]

# Perform the merge operation using pandas
final_merged_df = pd.merge(cdi_merged_df, filtered_disease_counts,
                           on=['LocationDesc', 'YearStart', 'Topic'], how='left')


# Display information about the final DataFrame for verification
final_merged_df.info()


# Assuming 'data' is your DataFrame
df = final_merged_df


# --------------#
# Visualization #
# --------------#


# -----------------------------------
# Visualization 1: Bar Chart for Top 5 Cities by Average Incidence Rates
# -----------------------------------

# Calculate the average incidence rate for each combination of LocationAbbr and Topic
avg_incidence = df.groupby(['LocationAbbr', 'Topic'])['IncidenceRate'].mean().reset_index()

# Find the top 5 cities with the highest average incidence rates
top_cities = avg_incidence.sort_values(by='IncidenceRate', ascending=False)\
                .drop_duplicates(subset=['LocationAbbr'])\
                .head(5)['LocationAbbr']

# Filter the avg_incidence DataFrame to only include those top 5 cities
top_cities_avg_incidence = avg_incidence[avg_incidence['LocationAbbr'].isin(top_cities)]

# Create a larger figure size for the bar chart
plt.figure(figsize=(14, 8))

# Bar plot with LocationAbbr on the x-axis and IncidenceRate on the y-axis, colored by Topic
sns.barplot(data=top_cities_avg_incidence, x='LocationAbbr', y='IncidenceRate', hue='Topic')

# Enhancing the chart with labels and title
plt.xlabel('Location Abbreviation', fontsize=12)
plt.ylabel('Average Incidence Rate', fontsize=12)
plt.title('Top 5 Cities: Average Incidence Rates by Location and Topic', fontsize=14)
plt.xticks(rotation=45)  # Rotate x-axis labels for readability

# Display the bar chart
plt.tight_layout()
plt.show()

# -----------------------------------
# Visualization 2: Nested Pie Chart for Topic and StratificationID1 Distribution
# -----------------------------------

# Calculate counts for 'Topic' and 'StratificationID1'
topic_counts = df['Topic'].value_counts()
stratification_counts = df['StratificationID1'].value_counts()

# Setup for the nested pie chart
plt.figure(figsize=(10, 7))

# Define colors for the slices
colors_outer = sns.color_palette('pastel', len(stratification_counts))
colors_inner = sns.color_palette('bright', len(topic_counts))

# Outer Pie Chart for StratificationID1
plt.pie(stratification_counts, radius=3, colors=colors_outer, autopct='%1.1f%%', pctdistance=0.86, startangle=140)

# Inner Pie Chart for Topic
plt.pie(topic_counts, radius=2, colors=colors_inner, autopct='%1.1f%%', pctdistance=0.75, startangle=140)

# Convert to a donut chart by adding a white circle in the center
centre_circle = plt.Circle((0,0),0.6, color='black', fc='white', linewidth=0)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

# Ensure the pie chart is drawn as a circle
plt.axis('equal')
plt.tight_layout()
plt.title('Nested Pie Chart: Topic and StratificationID1 Distribution')

# Add a legend outside the plot for clarity
plt.legend(loc='center left', labels=['Outer: ' + s for s in stratification_counts.index] + ['Inner: ' + t for t in topic_counts.index], bbox_to_anchor=(1, 0.5))

# Display the nested pie chart
plt.show()

# Sampling a fraction of the data for PCA analysis to make computations more manageable
final_merged_df = final_merged_df.sample(frac=0.05, random_state=42)

# --------- Data Preprocessing for PCA ---------
# Drop columns that are not relevant for PCA analysis to simplify the dataset
pca_df = final_merged_df.drop([
    'YearStart', 'YearEnd', 'LocationDesc', 'StratificationCategoryID1', 
    'Question', 'DataValue', 'LowConfidenceLimit', 'HighConfidenceLimit', 
    'GeoLocation', 'LocationID'
], axis=1)

# Check the structure of the dataframe after column removal
pca_df.info()

# List all columns in the dataframe to ensure correct columns are dropped
print(pca_df.columns.tolist())

# Checking for missing values in the dataset to ensure data quality
missing_values = pca_df.isnull().sum()

# Identifying categorical columns for encoding since PCA requires numerical input
categorical_columns = pca_df.select_dtypes(include=['object']).columns

# Display missing values and categorical columns to assess data readiness for PCA
print(missing_values, categorical_columns.tolist())

# Encoding categorical variables to convert them into a numeric format for PCA
pca_df_encoded = pd.get_dummies(pca_df, columns=['LocationAbbr', 'Topic', 'StratificationID1'])

# Display the first few rows of the cleaned and encoded DataFrame
print(pca_df_encoded.head())

# --------- Feature Scaling ---------
# Initializing the StandardScaler to normalize the features
scaler = StandardScaler()

# Scaling the features to have a mean of 0 and a standard deviation of 1
scaled_features = scaler.fit_transform(pca_df_encoded)

# Displaying the first few scaled features to verify scaling
print(scaled_features[:5])

# --------- Principal Component Analysis (PCA) ---------
# Initializing PCA without specifying the number of components to retain all components initially
pca_full = PCA()

# Fitting PCA on the scaled features
pca_full.fit(scaled_features)

# Calculating the cumulative explained variance ratio to understand how many components are needed
explained_variance_ratio = pca_full.explained_variance_ratio_
cumulative_explained_variance = explained_variance_ratio.cumsum()

# Plotting the cumulative explained variance to determine the optimal number of PCA components
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--')
plt.title('Cumulative Explained Variance by PCA Components')
plt.xlabel('Number of PCA Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.axhline(y=0.95, color='r', linestyle='-')  # 95% variance line for reference
plt.text(0.5, 0.85, '95% cut-off threshold', color = 'red', fontsize=16)

# Determining the number of components that explain at least 95% of the variance
optimal_num_components = len(cumulative_explained_variance[cumulative_explained_variance >= 0.95]) + 1

# Highlighting the optimal number of components on the plot
plt.axvline(x=optimal_num_components, color='g', linestyle='--')
plt.text(optimal_num_components + 1, 0.6, f'Optimal Components: {optimal_num_components}', color = 'green', fontsize=14)

# Displaying the plot
plt.show()


# Assuming 'scaled_features' is your pre-processed and scaled dataset ready for PCA

# Specify the number of components determined to be optimal from previous analysis
n_components = 12  # This number was identified from the cumulative explained variance plot

# --------- Applying PCA with Optimal Number of Components ---------
# Initialize PCA with the determined optimal number of components
pca = PCA(n_components=n_components)
# Fit PCA on the scaled features and transform the data
pca_result = pca.fit_transform(scaled_features)

# --------- Visualizing PCA Results ---------
# Creating a 2D plot to visualize the first two PCA components
plt.figure(figsize=(10, 8))
plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D PCA Results')
plt.grid(True)
plt.show()

# --------- Determining Optimal Number of Clusters with Elbow Method ---------
# Initialize an empty list to hold the Within-Cluster Sum of Squares (WCSS) for different cluster numbers
wcss = []

# Loop over a range of cluster numbers to find the optimal one
for i in range(1, 12):  # Trying different numbers of clusters
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(pca_result)
    wcss.append(kmeans.inertia_)  # Inertia: Sum of squared distances of samples to their closest cluster center

# Plotting the Elbow Method graph to identify the 'elbow' point
plt.figure(figsize=(10, 6))
plt.plot(range(1, 12), wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Determining the Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

# --------- Applying KMeans Clustering with Optimal Number of Clusters ---------
# Based on the Elbow Method plot, determine the appropriate number of clusters (manually identified here)
n_clusters = 8

# Initialize and fit KMeans with the determined number of clusters
kmeans_pca = KMeans(n_clusters=n_clusters, random_state=42)
kmeans_pca.fit(pca_result)

# Retrieve the cluster labels for each data point
cluster_labels = kmeans_pca.labels_

# --------- Visualizing KMeans Clusters on PCA-Reduced Data ---------
# Create a figure for plotting
plt.figure(figsize=(10, 8))

# Loop over each cluster number and plot the data points belonging to that cluster
for i in range(n_clusters):
    # Select data points that belong to the cluster 'i'
    ds = pca_result[cluster_labels == i]
    # Plot these data points with a unique label
    plt.scatter(ds[:, 0], ds[:, 1], label=f'Cluster {i+1}', alpha=0.5)

# Plotting the centroids of the clusters
centers = kmeans_pca.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], s=300, c='black', label='Centers', marker='*')

plt.title('KMeans Clusters Visualization on PCA-Reduced Data')
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.legend()
plt.show()


# Assuming 'pca' is your PCA model and 'pca_df_encoded' is your pre-processed data ready for PCA
# Also assuming 'kmeans_pca' is your fitted KMeans model on PCA-reduced data

# --------- PCA Component Loadings ---------
# Extract PCA components (loadings)
pca_components = pca.components_

# Convert PCA loadings to a DataFrame for better visualization and analysis
# Use feature names as columns for clear identification
pca_loadings_df = pd.DataFrame(pca_components, columns=pca_df_encoded.columns,
                               index=[f'PC{i+1}' for i in range(n_components)])

# Display the PCA loadings DataFrame
print(pca_loadings_df)

# --------- Visualizing PCA Loadings with Heatmap ---------
# Plot a heatmap of the PCA loadings to visualize contribution of each feature to the components
plt.figure(figsize=(12, 6))
sns.heatmap(pca_loadings_df, cmap="YlGnBu", annot=False)  # 'annot=True' for annotations, if needed
plt.title('PCA Loadings')
plt.show()

# --------- Inverse Transforming Cluster Centers to Original Feature Space ---------
# Inverse transform the KMeans cluster centers from PCA space to the original feature space
# This helps interpret the cluster centers in terms of original features
original_space_centroids = scaler.inverse_transform(pca.inverse_transform(kmeans_pca.cluster_centers_))

# Create a DataFrame for the inverse-transformed cluster centers for easier analysis
centroids_df = pd.DataFrame(original_space_centroids, columns=pca_df_encoded.columns)

# Compare the centroids with the mean of the original data
original_means = pca_df_encoded.mean(axis=0)  # Mean of original data before PCA
centroids_comparison_df = centroids_df.copy()
centroids_comparison_df.loc['Mean'] = original_means  # Append mean to the centroids DataFrame

# Display the comparison DataFrame
print(centroids_comparison_df)

# --------- Identifying Top Contributing Features for Each Principal Component ---------
# After fitting PCA, extract the absolute values of the PCA loadings to focus on magnitude of contribution
pca_loadings_analysis = pd.DataFrame(np.abs(pca.components_), columns=pca_df_encoded.columns,
                                     index=[f'PC{i+1}' for i in range(pca.n_components)])

# For each principal component, identify the top 5 features based on their loadings
top_features_per_pc = pca_loadings_analysis.apply(lambda s: s.nlargest(5).index.tolist(), axis=1)

# Display the top contributing features for each principal component
print(top_features_per_pc)



