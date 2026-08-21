import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Load dataset
df = pd.read_csv("data/creditcard.csv")

# Separate features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Create separate scalers
time_scaler = StandardScaler()
amount_scaler = StandardScaler()

# Scale Time
X["Time"] = time_scaler.fit_transform(X[["Time"]])

# Scale Amount
X["Amount"] = amount_scaler.fit_transform(X[["Amount"]])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Random Forest model...")

# Create model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

# Train model
model.fit(X_train, y_train)

print("Model training completed!")

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluation
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

# Save model
joblib.dump(model, "models/fraud_model.pkl")

# Save scalers separately
joblib.dump(time_scaler, "models/time_scaler.pkl")
joblib.dump(amount_scaler, "models/amount_scaler.pkl")

print("\nModel saved to models/fraud_model.pkl")
print("Time scaler saved to models/time_scaler.pkl")
print("Amount scaler saved to models/amount_scaler.pkl")