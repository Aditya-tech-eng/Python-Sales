# Python-Sales
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, accuracy_score
import tensorflow as tf
from tensorflow import keras

# Load dataset
# Using 'unicode_escape' encoding to avoid character encoding issues
df = pd.read_csv('Diwali Sales Data.csv', encoding='unicode_escape')

# Display dataset shape (rows, columns)
print("Dataset Shape:", df.shape)

# Display first few rows to inspect data
print(df.head())

# Data Information
df.info()

# Drop unrelated/blank columns
df.drop(['Status', 'unnamed1'], axis=1, inplace=True)

# Check for null values
print("Null values:")
print(pd.isnull(df).sum())

# Drop rows with null values
df.dropna(inplace=True)

# Convert Amount column to integer type
df['Amount'] = df['Amount'].astype(int)

# Rename 'Marital_Status' column
df.rename(columns={'Marital_Status': 'Shaadi'}, inplace=True)

# Statistical summary of numerical columns
print(df.describe())

# Exploratory Data Analysis

# Gender Distribution
plt.figure(figsize=(6,4))
ax = sns.countplot(x='Gender', data=df)
plt.title("Gender Distribution")
for bars in ax.containers:
    ax.bar_label(bars)
plt.show()

# Gender vs Total Purchase Amount
sales_gen = df.groupby('Gender', as_index=False)['Amount'].sum().sort_values(by='Amount', ascending=False)
plt.figure(figsize=(6,4))
sns.barplot(x='Gender', y='Amount', data=sales_gen)
plt.title("Total Purchase Amount by Gender")
plt.show()

# Age Group Analysis
plt.figure(figsize=(6,4))
ax = sns.countplot(x='Age Group', hue='Gender', data=df)
plt.title("Age Group Distribution")
for bars in ax.containers:
    ax.bar_label(bars)
plt.show()

# Total Amount vs Age Group
sales_age = df.groupby('Age Group', as_index=False)['Amount'].sum().sort_values(by='Amount', ascending=False)
plt.figure(figsize=(6,4))
sns.barplot(x='Age Group', y='Amount', data=sales_age)
plt.title("Total Purchase Amount by Age Group")
plt.show()

# Machine Learning Models

# Data Preprocessing
# Select relevant features for predictive modeling
features = ['Age', 'Orders', 'Amount']
X = df[features]

y_regression = df['Amount']  # Target variable for regression
y_classification = (df['Amount'] > df['Amount'].median()).astype(int)  # Binary classification (high/low spending)

# Train-Test Split
X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_regression, test_size=0.2, random_state=42)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X, y_classification, test_size=0.2, random_state=42)

# Standardizing Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled_clf = scaler.fit_transform(X_train_clf)
X_test_scaled_clf = scaler.transform(X_test_clf)

# Linear Regression Model
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train_reg)
y_pred_reg = lin_reg.predict(X_test_scaled)
print("Linear Regression MSE:", mean_squared_error(y_test_reg, y_pred_reg))

# Random Forest Regression
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train_reg)
y_pred_rf_reg = rf_reg.predict(X_test)
print("Random Forest Regression MSE:", mean_squared_error(y_test_reg, y_pred_rf_reg))

# Logistic Regression for Classification
log_reg = LogisticRegression()
log_reg.fit(X_train_scaled_clf, y_train_clf)
y_pred_clf = log_reg.predict(X_test_scaled_clf)
print("Logistic Regression Accuracy:", accuracy_score(y_test_clf, y_pred_clf))

# Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_clf, y_train_clf)
y_pred_rf_clf = rf_clf.predict(X_test_clf)
print("Random Forest Classification Accuracy:", accuracy_score(y_test_clf, y_pred_rf_clf))

# K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
df['Cluster'] = kmeans.labels_
print("Customer Segmentation Clusters Assigned:")
print(df[['User_ID', 'Cluster']].head())

# Deep Learning Model
model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train_scaled, y_train_reg, epochs=20, verbose=1)
y_pred_dl = model.predict(X_test_scaled)
print("Neural Network MSE:", mean_squared_error(y_test_reg, y_pred_dl))

print("ML models successfully executed!")
