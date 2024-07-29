### <div align="center"> <h1> DataSegmentation </h1> </div>

Welcome to the Segmentation Project! This repository contains all the necessary code and documentation for performing data segmentation, an essential step in data analysis and machine learning that involves dividing data into distinct groups or segments based on specific characteristics. This project aims to provide an efficient and robust method for segmenting datasets to enable better analysis and insights.


### Overview
This project is designed to provide a comprehensive approach to data segmentation, which is a critical task in data analysis and machine learning. Segmentation involves dividing a dataset into distinct groups or segments that share similar characteristics. This can help in uncovering patterns, making informed decisions, and tailoring strategies to different segments.

### Objectives
The objective of the Segmentation Project is to provide a comprehensive framework for dividing a dataset into meaningful and distinct segments. This is achieved by implementing and comparing various segmentation algorithms, visualizing the results, and evaluating the effectiveness of each approach. The project aims to facilitate deeper data analysis and provide actionable insights by enabling users to identify and understand patterns within their data.

### Dataset

- Source: CC GENERAL.csv
- Size: 8950 records and 18 variables.

  ## Let's Get Started :

  **First of all Import all the necessary modules in the Jupyter Notebook.**

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import scipy.stats as stats

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler

**Now Import the Dataset.**

DA = pd.read_csv('CC GENERAL.csv')

![image](https://github.com/user-attachments/assets/2bc3867a-4437-4cdf-98ed-c1eabef91f72)

DA is our DataFrame Name. There are 8950 records and 18 variables in our DataFrame

**Using `DA.describe()`**

![image](https://github.com/user-attachments/assets/b2bc2591-6f7f-465d-a6a9-58415901e476)

`DA.describe()` is a method used to generate a statistical summary of the dataset, providing key metrics such as mean, standard deviation, and percentiles for numerical columns. This function is crucial for understanding the basic distribution and characteristics of the data before performing segmentation.


### Checking if there are any Missing Values

DA.isna().sum()

![image](https://github.com/user-attachments/assets/c47a6fed-ad38-4372-ad3b-708c80b8436d)

### Filling the missing value with median

DA['CREDIT_LIMIT'] = DA.CREDIT_LIMIT.fillna(DA['CREDIT_LIMIT'].median())

DA['MINIMUM_PAYMENTS'] = DA.MINIMUM_PAYMENTS.fillna(DA['MINIMUM_PAYMENTS'].median())

![image](https://github.com/user-attachments/assets/707432cb-347d-40ea-89ad-2b4d3bc7089f)

### Now copy the original DataFrame and drop the Cust_Id column as it will not give so much impact.

DA_copy = DA_copy.drop(columns=['CUST_ID'])

![image](https://github.com/user-attachments/assets/089bfb38-4ef4-4ae7-8a47-b87c13ed05b5)


## Feature Engineering

### Dimenasionality Reduction

**PCA : Principle Component Analysis**


Because y variable is not present so we cannot do correlational analysis to do feature engineering or Dimenasionality Reduction, here in unsupervised ML algorithms, PCA Principle Component Analysis is used.

pc = PCA(n_components = 17).fit(DA_copy_scaled)

pc.explained_variance_

- Initialization and Fit: We initialized the PCA model to reduce the dataset to 17 components and fit it to the standardized data.

sum(pc.explained_variance_)

pc.explained_variance_ /  sum(pc.explained_variance_)

pc.explained_variance_ratio_

- Explained Variance: We examined the variance explained by each principal component and the proportion of total variance they explain.

var = np.round(np.cumsum(pc.explained_variance_ratio_) * 100, 2)

pd.DataFrame({'Eigen_Values':pc.explained_variance_,
                   
                   'VAR':np.round(pc.explained_variance_ratio_*100,2),
                     
                     'CUM_VAR':var},index=range(1,18))

- Cumulative Variance: We calculated the cumulative variance explained by the components to understand how many components capture a significant portion of the total variance.

pc_final=PCA(n_components=6).fit(DA_copy_scaled)

pc_final.explained_variance_

- Component Selection: We selected 6 principal components based on cumulative variance and refitted the PCA model.

reduced_cr=pc_final.transform(DA_copy_scaled)

dimensions=pd.DataFrame(reduced_cr)

dimensions.columns=['C1','C2','C3','C4','C5','C6']

- Data Transformation: We transformed the data to the 6 principal components, creating a new DataFrame for these reduced dimensions.


### Clusterings

Clustering is a method in data analysis where similar items are grouped together into clusters. It helps find natural groupings or patterns in data without needing predefined labels. This technique is useful for organizing data, identifying similarities, and exploring underlying structures in datasets.

km_4 = KMeans( n_clusters = 4, random_state = 123).fit( dimensions )

km_5 = KMeans( n_clusters = 5, random_state = 123).fit( dimensions )

km_6 = KMeans( n_clusters = 6, random_state = 123).fit( dimensions )

km_7 = KMeans( n_clusters = 7, random_state = 123).fit( dimensions )

km_8 = KMeans( n_clusters = 8, random_state = 123).fit( dimensions )

km_9 = KMeans( n_clusters = 9, random_state = 123).fit( dimensions )

km_10 = KMeans( n_clusters = 10, random_state = 123).fit( dimensions )


- Initialized and fit KMeans clustering models with different numbers of clusters (from 3 to 10) on the reduced dimensions (dimensions DataFrame).


DA_copy_scaled['cluster_3'] = km_3.labels_

DA_copy_scaled['cluster_4'] = km_4.labels_

DA_copy_scaled['cluster_5'] = km_5.labels_

DA_copy_scaled['cluster_6'] = km_6.labels_

DA_copy_scaled['cluster_7'] = km_7.labels_

DA_copy_scaled['cluster_8'] = km_8.labels_

DA_copy_scaled['cluster_9'] = km_9.labels_

DA_copy_scaled['cluster_10'] = km_10.labels_


- Saved cluster labels for all models (from 3 to 10 clusters) and appended them to the DA_copy_scaled DataFrame for further analysis.

![image](https://github.com/user-attachments/assets/bf272d96-ec0c-41b5-8233-68bdbef60385)


## Quantitative evaluation of number of clusters

### 1. Analysis of the cluster size | Count method

pd.Series(km_3.labels_).value_counts()

pd.Series(km_3.labels_).value_counts()/sum(pd.Series(km_3.labels_).value_counts())

pd.Series(km_4.labels_).value_counts()/sum(pd.Series(km_4.labels_).value_counts())

pd.Series(km_5.labels_).value_counts()/sum(pd.Series(km_5.labels_).value_counts())

pd.Series(km_6.labels_).value_counts()/sum(pd.Series(km_6.labels_).value_counts())

pd.Series(km_7.labels_).value_counts()/sum(pd.Series(km_7.labels_).value_counts())

pd.Series(km_8.labels_).value_counts()/sum(pd.Series(km_8.labels_).value_counts())

pd.Series(km_9.labels_).value_counts()/sum(pd.Series(km_9.labels_).value_counts())

pd.Series(km_10.labels_).value_counts()/sum(pd.Series(km_10.labels_).value_counts())

- Each line calculates the number of data points assigned to each cluster by km, where km is one of the KMeans models.

- Dividing by the total count provides the percentage distribution of data points across clusters.

![image](https://github.com/user-attachments/assets/397a526d-c432-40bc-9bfe-29133e90b263)

- Computes the sum of squared distances of samples to their closest cluster center.

- Lower inertia values indicate tighter clusters, suggesting better-defined and more distinct clusters in the data.

### 2. Elbow Analysis

for num_clusters in cluster_range:
    
    clusters = KMeans( num_clusters ).fit( dimensions )
    
    cluster_errors.append( clusters.inertia_ )

clusters_df = pd.DataFrame( { "num_clusters": cluster_range, "cluster_errors": cluster_errors } )

clusters_df.head(5)

- Automating the process of finding the optimal number of clusters for KMeans clustering using the elbow method. It calculates and plots the inertia values for different numbers of clusters, enabling data-driven decision-making on the appropriate K value for clustering the dimensions data.

%matplotlib inline

import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))

plt.plot( clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o" )

plt.show()

![image](https://github.com/user-attachments/assets/727b0633-607e-416b-973a-6411ea0d5ac7)

DA_copy_scaled.cluster_5.value_counts()/sum(DA_copy_scaled.cluster_5.value_counts())

![image](https://github.com/user-attachments/assets/b52d7392-f1f4-4998-8ffd-c9a85e25701d)


### 3. Choosing number clusters using Silhouette Coefficient -- SC


metrics.silhouette_score( dimensions, km_4.labels_ )

k_range = range(3, 18)

scores = []

for k in k_range:
    
    km = KMeans(n_clusters = k, random_state = 123)
    
    km.fit( dimensions )
    
    scores.append( metrics.silhouette_score(dimensions, km.labels_) )


- This is used to assess the quality of clustering results using the Silhouette Coefficient across different numbers of clusters.

- The Silhouette Coefficient helps in identifying the optimal number of clusters by indicating how well-separated the clusters are.


plt.plot(k_range, scores, marker = "o")

plt.xlabel('Number of clusters')

plt.ylabel('Silhouette Coefficient')

plt.grid(True)

![image](https://github.com/user-attachments/assets/4c067c18-2080-429a-840a-ac4ded370a02)

- The plotted graph (Number of clusters vs Silhouette Coefficient) visually identifies the K value that maximizes clustering performance for the given data

## Segment Distribution

![image](https://github.com/user-attachments/assets/bdca41e5-2f70-43a9-8dcd-d33a9d8d7e1c)







































































































