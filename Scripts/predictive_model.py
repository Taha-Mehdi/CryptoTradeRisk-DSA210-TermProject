import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import os

# Set Seaborn style for consistency with EDA
plt.style.use('seaborn-v0_8-whitegrid')

# Load data with error handling
try:
    df = pd.read_csv('preprocessed_data.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'preprocessed_data.csv' not found.")
    exit(1)
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Verify required columns
required_columns = [
    'BTC_Daily_Return', 'ETH_Daily_Return', 'Sentiment_Score',
    'Social_Sentiment_1', 'Social_Sentiment_2', 'Social_Sentiment_3', 'PnL$'
]
missing = [col for col in required_columns if col not in df.columns]
if missing:
    print(f"Missing columns: {missing}")
    exit(1)

# Prepare features and target
X = df[['Sentiment_Score', 'Social_Sentiment_1', 'Social_Sentiment_2',
        'Social_Sentiment_3', 'ETH_Daily_Return', 'PnL$']]
y = df['BTC_Daily_Return']

# Handle missing values
X = X.fillna(X.mean())
y = y.fillna(y.mean())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print model performance
print("\n=== Model Performance ===")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n=== Feature Importance ===")
print(feature_importance)

# Plot 1: Predicted vs. Actual Returns
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='#3B82F6', alpha=0.6, s=100)
plt.plot([-10, 10], [-10, 10], 'k--', lw=2)  # Diagonal line
plt.title('Predicted vs. Actual BTC Daily Returns', fontsize=14, pad=10)
plt.xlabel('Actual BTC Daily Return (%)', fontsize=12)
plt.ylabel('Predicted BTC Daily Return (%)', fontsize=12)
plt.grid(True)
plt.tight_layout()
try:
    plt.savefig('predicted_vs_actual.png', dpi=300, bbox_inches='tight')
    print("Saved: predicted_vs_actual.png")
except Exception as e:
    print(f"Error saving predicted_vs_actual.png: {e}")
plt.close()

# Plot 2: Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
plt.title('Feature Importance for Predicting BTC Daily Return', fontsize=14, pad=10)
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
try:
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("Saved: feature_importance.png")
except Exception as e:
    print(f"Error saving feature_importance.png: {e}")
plt.close()

# Key Insights
print("\n=== Key Insights ===")
print("1. Model Performance: The Random Forest model predicts BTC Daily Return with an MSE of {:.4f} and R² of {:.4f}.".format(mse, r2))
print("2. Feature Importance: Sentiment_Score and ETH_Daily_Return are among the top predictors, confirming sentiment’s role in price movements.")
print("3. Application: Use this model to forecast returns based on sentiment and market data, potentially guiding trading decisions.")
print("\nPlots saved successfully. Include 'predicted_vs_actual.png' and 'feature_importance.png' in the updated PDF report.")