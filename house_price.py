import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the Boston Housing dataset using fetch_openml
boston = fetch_openml(name="boston", version=1, as_frame=True)

# Convert the dataset to a Pandas DataFrame
data = boston.frame
data.columns = data.columns.str.upper()  # Standardize column names for consistency
data.rename(columns={"MEDV": "PRICE"}, inplace=True)  # Rename the target column for clarity

# Ensure data is numeric
data = data.apply(pd.to_numeric, errors='coerce')

# Define features (X) and target (y)
X = data.drop(columns='PRICE')
y = data['PRICE']

# Split the dataset into training and testing sets
test_size = 0.2  # Modify test size if needed
random_state = 42  # Change for different random splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# Ensure training and test data are numeric arrays or DataFrames
X_train = X_train.values
X_test = X_test.values

# Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Predictions using Linear Regression
y_pred_linear = linear_model.predict(X_test)

# Evaluate Linear Regression Model
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

# Ridge Regression Model with multiple alpha values
alphas = [0.1, 1.0, 10.0, 50.0, 100.0]  # Test different alpha values
ridge_metrics = []

for alpha in alphas:
    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(X_train, y_train)
    y_pred_ridge = ridge_model.predict(X_test)
    mse_ridge = mean_squared_error(y_test, y_pred_ridge)
    r2_ridge = r2_score(y_test, y_pred_ridge)
    ridge_metrics.append((alpha, mse_ridge, r2_ridge))

# Print Evaluation Metrics
print("Linear Regression:")
print(f"Mean Squared Error: {mse_linear:.2f}")
print(f"R^2 Score: {r2_linear:.2f}")

print("\nRidge Regression (with varying alpha):")
for alpha, mse, r2 in ridge_metrics:
    print(f"Alpha: {alpha:.1f} | Mean Squared Error: {mse:.2f} | R^2 Score: {r2:.2f}")

# Visualization: Compare metrics for Linear and Ridge regression
metrics = ['MSE', 'R^2 Score']
x = np.arange(len(metrics))
width = 0.2  # Bar width

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width, [mse_linear, r2_linear], width, label='Linear Regression', color='blue')

for i, (alpha, mse, r2) in enumerate(ridge_metrics):
    rects_ridge = ax.bar(x + (i * width), [mse, r2], width, label=f'Ridge (alpha={alpha})')

ax.set_xlabel('Metrics')
ax.set_ylabel('Scores')
ax.set_title('Model Performance Metrics')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
plt.tight_layout()
plt.show()

# Heatmap of Feature Correlations
plt.figure(figsize=(12, 8))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Heatmap of Feature Correlations')
plt.show()

# Visualization of predictions
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred_linear, color='blue', label='Linear Regression Predictions')
for i, (alpha, mse, r2) in enumerate(ridge_metrics):
    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(X_train, y_train)
    y_pred_ridge = ridge_model.predict(X_test)
    plt.scatter(y_test, y_pred_ridge, label=f'Ridge Predictions (alpha={alpha})')
    
plt.plot([0, 50], [0, 50], color='black', linestyle='--', label='Perfect Fit Line')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.legend()
plt.show()
