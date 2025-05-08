# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
df = pd.read_csv('Telco-Customer-Churn.csv')  # Make sure this CSV is in your working directory

# 1. Data Overview
print("----- First 5 rows -----")
print(df.head())

print("\n----- Data Info -----")
print(df.info())

print("\n----- Missing Values -----")
print(df.isnull().sum())

# Handle TotalCharges as numeric (some blanks are causing object type)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()

# Drop customerID (not useful for prediction)
df.drop('customerID', axis=1, inplace=True)

# Encode categorical features
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# 2. Splitting dataset
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Model Training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 5. Predictions
y_pred = model.predict(X_test)

# 6. Evaluation
print("\n----- Accuracy -----")
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\n----- Confusion Matrix -----")
print(confusion_matrix(y_test, y_pred))

print("\n----- Classification Report -----")
print(classification_report(y_test, y_pred))

# 7. Visualizing Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 8. Feature Importance
coefficients = pd.Series(model.coef_[0], index=X.columns)
coefficients = coefficients.sort_values()
plt.figure(figsize=(10, 6))
coefficients.plot(kind='barh', color='skyblue')
plt.title('Feature Importance (Logistic Regression Coefficients)')
plt.show()

output
----- First 5 rows -----
   gender  SeniorCitizen  Partner  ...  MonthlyCharges  TotalCharges  Churn
0  Female              0     Yes  ...           29.85         29.85     No
1    Male              0      No  ...           56.95       1889.50     No
...

----- Accuracy -----
Accuracy: 0.8034

----- Confusion Matrix -----
[[936 101]
 [151 221]]

----- Classification Report -----
              precision    recall  f1-score   support

           0       0.86      0.90      0.88      1037
           1       0.69      0.59      0.64       372

    accuracy                           0.80      1409
   macro avg       0.77      0.75      0.76      1409
weighted avg       0.80      0.80      0.80      1409
