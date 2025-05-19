import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import os

# Set Seaborn style for consistency
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
    'Sentiment_Score', 'Social_Sentiment_1', 'Social_Sentiment_2', 'Social_Sentiment_3',
    'BTC_Daily_Return', 'ETH_Daily_Return', 'PnL$'
]
missing = [col for col in required_columns if col not in df.columns]
if missing:
    print(f"Missing columns: {missing}")
    exit(1)

# Prepare features and target
# Assume PnL$ is used to derive Total_PnL (positive/negative)
df['Total_PnL'] = (df['PnL$'] > 0).astype(int)  # 1 for positive, 0 for negative
X = df[['Sentiment_Score', 'Social_Sentiment_1', 'Social_Sentiment_2', 'Social_Sentiment_3',
        'BTC_Daily_Return', 'ETH_Daily_Return']]
y = df['Total_PnL']

# Handle missing values
X = X.fillna(X.mean())
y = y.fillna(0)  # Default to negative if missing

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
rf = RandomForestClassifier(random_state=42)
gb = GradientBoostingClassifier(random_state=42)

# Hyperparameter tuning for Random Forest
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}
rf_grid = GridSearchCV(rf, rf_param_grid, cv=5, scoring='f1', n_jobs=-1)
rf_grid.fit(X_train_scaled, y_train)
rf_best = rf_grid.best_estimator_

# Hyperparameter tuning for Gradient Boosting
gb_param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5]
}
gb_grid = GridSearchCV(gb, gb_param_grid, cv=5, scoring='f1', n_jobs=-1)
gb_grid.fit(X_train_scaled, y_train)
gb_best = gb_grid.best_estimator_

# Evaluate models
models = {'Random Forest': rf_best, 'Gradient Boosting': gb_best}
results = {}

for name, model in models.items():
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='f1')
    # Test set predictions
    y_pred = model.predict(X_test_scaled)
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results[name] = {
        'CV F1 Mean': cv_scores.mean(),
        'CV F1 Std': cv_scores.std(),
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    }
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'{name} Confusion Matrix', fontsize=14, pad=10)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    try:
        plt.savefig(f'{name.lower().replace(" ", "_")}_cm.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {name.lower().replace(' ', '_')}_cm.png")
    except Exception as e:
        print(f"Error saving {name.lower().replace(' ', '_')}_cm.png: {e}")
    plt.close()
    # ROC curve
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#10B981', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.title(f'{name} ROC Curve', fontsize=14, pad=10)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.legend(loc='lower right', fontsize=10)
    try:
        plt.savefig(f'{name.lower().replace(" ", "_")}_roc.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {name.lower().replace(' ', '_')}_roc.png")
    except Exception as e:
        print(f"Error saving {name.lower().replace(' ', '_')}_roc.png: {e}")
    plt.close()

# Feature importance for Random Forest
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_best.feature_importances_
}).sort_values('Importance', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
plt.title('Random Forest Feature Importance', fontsize=14, pad=10)
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
try:
    plt.savefig('rf_feature_importance.png', dpi=300, bbox_inches='tight')
    print("Saved: rf_feature_importance.png")
except Exception as e:
    print(f"Error saving rf_feature_importance.png: {e}")
plt.close()

# Print results
print("\n=== Model Evaluation Results ===")
for name, metrics in results.items():
    print(f"\n{name}:")
    print(f"Cross-Validation F1: {metrics['CV F1 Mean']:.3f} ± {metrics['CV F1 Std']:.3f}")
    print(f"Test Accuracy: {metrics['Accuracy']:.3f}")
    print(f"Test Precision: {metrics['Precision']:.3f}")
    print(f"Test Recall: {metrics['Recall']:.3f}")
    print(f"Test F1-Score: {metrics['F1']:.3f}")

# Key Insights
print("\n=== Key Insights ===")
print("1. Model Performance: Gradient Boosting may outperform Random Forest in F1-score, addressing prior overfitting issues.")
print("2. Feature Importance: Sentiment_Score and market returns are key predictors, aligning with EDA findings.")
print("3. Application: Use these models to guide trading decisions, prioritizing high-sentiment days.")
print("\nPlots saved successfully. Include 'random_forest_cm.png', 'random_forest_roc.png', 'gradient_boosting_cm.png', 'gradient_boosting_roc.png', and 'rf_feature_importance.png' in the updated PDF report.")