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























































































































































































