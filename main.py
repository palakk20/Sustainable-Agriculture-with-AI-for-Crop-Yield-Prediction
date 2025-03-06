import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load datasets
yield_df = pd.read_csv('yield_df.csv')
rainfall = pd.read_csv('rainfall.csv')
temp = pd.read_csv('temp.csv')
pesticides = pd.read_csv('pesticides.csv')

# Check column names to debug missing columns
print("Yield DataFrame Columns:", yield_df.columns)
print("Rainfall DataFrame Columns:", rainfall.columns)
print("Temperature DataFrame Columns:", temp.columns)
print("Pesticides DataFrame Columns:", pesticides.columns)

# Standardize column names (remove spaces and ensure consistency)
yield_df.columns = yield_df.columns.str.strip()
rainfall.columns = rainfall.columns.str.strip()
temp.columns = temp.columns.str.strip()
pesticides.columns = pesticides.columns.str.strip()

# Rename columns for consistency
temp.rename(columns={'year': 'Year', 'country': 'Area'}, inplace=True)
rainfall.rename(columns={' Area': 'Area'}, inplace=True)
pesticides.rename(columns={'Value': 'Pesticide_Use'}, inplace=True)

# Merge datasets on common columns
data = yield_df.merge(rainfall, on=['Year', 'Area'], how='left')
data = data.merge(temp, on=['Year', 'Area'], how='left')
data = data.merge(pesticides, on=['Year', 'Area'], how='left')

# Handle missing values
data.ffill(inplace=True)

# Drop non-numeric columns
non_numeric_cols = data.select_dtypes(include=['object']).columns
data = data.drop(columns=non_numeric_cols)

# Feature selection - Drop only existing columns
drop_cols = [col for col in ['hg/ha_yield', 'Element', 'Domain', 'Unit'] if col in data.columns]
X = data.drop(columns=drop_cols)  # Features
y = data['hg/ha_yield'] if 'hg/ha_yield' in data.columns else data.iloc[:, -1]  # Target variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'MSE: {mse}')
print(f'R^2 Score: {r2}')

# Visualization 1: Feature importance
importances = model.feature_importances_
feature_names = X.columns
plt.figure(figsize=(10, 5))
sns.barplot(x=importances, y=feature_names)
plt.title('Feature Importance')
plt.show()

# Visualization 2: Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()

# Visualization 3: Yield trend over the years
plt.figure(figsize=(10, 5))
sns.lineplot(x=data['Year'], y=data['hg/ha_yield'])
plt.title('Crop Yield Trend Over Years')
plt.xlabel('Year')
plt.ylabel('Yield (hg/ha)')
plt.show()

# Insights & Sustainability Focus
print("\nSustainability Insights:")
print("- Optimize water usage based on rainfall trends.")
print("- Reduce excessive pesticide use to protect soil quality.")
print("- Choose climate-resilient crops for better yield.")
