import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

# Load and prepare dataset
df = pd.read_csv('preprocessed_data.csv')
df['Total_PnL'] = (df['PnL$'] > 0).astype(int)

features = [
    'Sentiment_Score', 'Social_Sentiment_1', 'Social_Sentiment_2',
    'Social_Sentiment_3', 'BTC_Daily_Return', 'ETH_Daily_Return'
]
X = df[features].fillna(df[features].mean())
y = df['Total_PnL'].fillna(0)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Compute class-balanced sample weights
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# Define models
models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=100, max_depth=None, class_weight='balanced', random_state=42
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
    )
}

# Evaluation for each model
for name, model in models.items():
    print(f"\nModel: {name}")

    # Fit model
    if name == 'Random Forest':
        model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
    else:
        model.fit(X_train_scaled, y_train)

    # Predictions and probabilities
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    # Accuracy and metrics
    acc = accuracy_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    report = classification_report(y_test, y_pred, target_names=["Loss", "Profit"])
    cm = confusion_matrix(y_test, y_pred)

    # Display results
    print("Accuracy:", round(acc, 3))
    print("ROC AUC:", round(roc_auc, 3))
    print("Classification Report:\n", report)

    # Save Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    cm_filename = f"{name.lower().replace(' ', '_')}_cm.png"
    plt.savefig(cm_filename)
    print(f"Saved: {cm_filename}")
    plt.close()

    # Save ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{name} - ROC Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    roc_filename = f"{name.lower().replace(' ', '_')}_roc.png"
    plt.savefig(roc_filename)
    print(f"Saved: {roc_filename}")
    plt.close()

    # Cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores = cross_val_score(model, X, y, scoring='f1', cv=skf)
    print(f"Cross-Validated F1 Score: {f1_scores.mean():.3f} ± {f1_scores.std():.3f}")