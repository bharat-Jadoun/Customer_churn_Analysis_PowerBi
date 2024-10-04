# 1. Data Loading and Cleaning
# First, you need to load the dataset and check for any missing or inconsistent values.

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv(r"C:\Users\DELL\Downloads\customer churn data.csv")

# Quick look at the data
print(df.info())

# Check for missing values
print(df.isnull().sum())

# Handle missing values (if any)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')  # Convert TotalCharges to numeric
df = df.dropna()  # Remove rows with missing values or use df.fillna() to fill missing values

# 2. Exploratory Data Analysis (EDA)
# a. Demographic Distribution
# You can analyze how many senior citizens, gender distribution, and other demographic factors exist.

# Gender distribution
sns.countplot(x='gender', data=df)
plt.title('Gender Distribution')
plt.show()

# Senior Citizens
sns.countplot(x='Senior Citizen', data=df)
plt.title('Senior Citizen Distribution')
plt.show()

# b. Churn Rate Analysis
# Analyze how many customers churned and visualize the churn rate.

# Churn distribution
sns.countplot(x='Churn', data=df)
plt.title('Churn Distribution')
plt.show()

# Churn rate by contract type
sns.countplot(x='Contract', hue='Churn', data=df)
plt.title('Churn by Contract Type')
plt.show()


# c. Tenure vs. Churn
# See if there’s a relationship between the length of time a customer has been with the company and their likelihood to churn.

# Boxplot of tenure against churn
sns.boxplot(x='Churn', y='tenure', data=df)
plt.title('Tenure vs. Churn')
plt.show()

# Tenure distribution based on churn
sns.histplot(data=df, x='tenure', hue='Churn', kde=True, bins=30)
plt.title('Tenure Distribution by Churn')
plt.show()

# 3. Correlation and Feature Importance
# You can calculate the correlation matrix to see how numeric features like MonthlyCharges and TotalCharges correlate with churn.

# Drop non-numeric columns for correlation analysis
# This includes columns like 'customerID' and other categorical data that are not needed for correlation
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Correlation matrix now
corr_matrix = numeric_df.corr()


# Plot the correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# 4. Churn Prediction: Feature Engineering and Model Building
# You can create a few new features and build a simple logistic regression model to predict churn.

# Feature engineering: Create a new feature for average monthly spend
df['AvgMonthlySpend'] = df['TotalCharges'] / df['tenure']

# Convert categorical variables into dummy variables
df = pd.get_dummies(df, columns=['gender', 'Contract', 'PaymentMethod', 'InternetService'], drop_first=True)

# Select features for the model
X = df[['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlySpend', 'gender_Male', 'Contract_Two year']]
y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a logistic regression model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate the model
from sklearn.metrics import classification_report, confusion_matrix
y_pred = model.predict(X_test)

# Print evaluation metrics
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 5. Visualizing the Churn Prediction Model Results
# You can visualize the model’s performance using a confusion matrix.

import seaborn as sns
from sklearn.metrics import confusion_matrix

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual Churn')
plt.xlabel('Predicted Churn')
plt.show()

# This code will allow you to:

# Clean the data and handle missing values.
# Explore the dataset to understand the key factors that might lead to customer churn.
# Build a basic logistic regression model to predict churn.
# Visualize important metrics and relationships within the data.
# Once you gather key insights from this, you can create dashboards in Power BI or Tableau to present the findings visually.





# The confusion matrix you have provided is a summary of prediction results for a binary classification problem 
# (in this case, predicting customer churn). Here's how to interpret it:

# Confusion Matrix:
# [[938  95]
#  [213 161]]
# True Positives (TP): 161 (bottom-right): The model correctly predicted churn (class 1) for 161 customers who actually churned.
# True Negatives (TN): 938 (top-left): The model correctly predicted no churn (class 0) for 938 customers who did not churn.
# False Positives (FP): 95 (top-right): The model incorrectly predicted churn (class 1) for 95 customers who actually did not churn (false alarms).
# False Negatives (FN): 213 (bottom-left): The model incorrectly predicted no churn (class 0) for 213 customers who actually churned (missed churns).
# Metrics Derived from the Confusion Matrix:
# Precision: The proportion of positive identifications that were actually correct.

# For class 0: 0.81
# For class 1: 0.63
# Interpretation: When the model predicted that a customer would churn, it was correct 63% of the time. However, for predicting non-churn (class 0), it was more accurate, with 81% precision.

# Recall (Sensitivity): The proportion of actual positives that were correctly identified by the model.

# For class 0: 0.91
# For class 1: 0.43
# Interpretation: The model identified 91% of non-churning customers correctly. However, it only correctly identified 43% of actual churners. This indicates that the model is missing a significant number of churn predictions.

# F1-Score: The harmonic mean of precision and recall, providing a balance between the two.

# For class 0: 0.86
# For class 1: 0.51
# Interpretation: The F1 score is much lower for class 1 (churners), which shows that the model struggles with churn prediction.

# Accuracy: The overall proportion of correct predictions.

# Accuracy: 0.78 (or 78%)
# Interpretation: The model correctly predicted whether a customer churned or not 78% of the time.

# Insights:
# Class Imbalance Issue: The recall for class 1 (churners) is quite low at 43%, meaning the model fails to identify a large proportion of actual churners. This could be due to class imbalance if there are significantly fewer churners than non-churners in the dataset.

# Good at Predicting Non-Churners: The model performs better in predicting non-churners (class 0) as evidenced by the high recall (91%) and precision (81%) for class 0.

# Improvement Needed for Churn Prediction: Since the business problem likely focuses on accurately predicting churn (class 1), the low recall and F1-score for class 1 indicates that the model may need improvement. This could be done by:

# Balancing the dataset (e.g., using techniques like SMOTE or undersampling).
# Adjusting the decision threshold.
# Using more advanced models (e.g., Random Forest, Gradient Boosting).