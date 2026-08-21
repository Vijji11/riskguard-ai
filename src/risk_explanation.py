import pandas as pd
import joblib
import shap
import numpy as np

# Load model
model = joblib.load("models/fraud_model.pkl")

# Load dataset
df = pd.read_csv("data/creditcard.csv")

# Prepare features
X = df.drop("Class", axis=1)

# Load scalers
time_scaler = joblib.load("models/time_scaler.pkl")
amount_scaler = joblib.load("models/amount_scaler.pkl")

# Scale Time and Amount
X["Time"] = time_scaler.transform(X[["Time"]])
X["Amount"] = amount_scaler.transform(X[["Amount"]])

# Create SHAP explainer
explainer = shap.TreeExplainer(model)


def explain_transaction(transaction):

    transaction_df = pd.DataFrame([transaction])

    # Scale Time
    transaction_df["Time"] = time_scaler.transform(
        transaction_df[["Time"]]
    )

    # Scale Amount
    transaction_df["Amount"] = amount_scaler.transform(
        transaction_df[["Amount"]]
    )

    # Get SHAP values
    shap_values = explainer.shap_values(transaction_df)

    # Convert to numpy
    shap_values = np.asarray(shap_values)

    print("SHAP output shape:", shap_values.shape)

    # Handle SHAP output formats
    if shap_values.ndim == 3:
        # Shape usually:
        # (number of transactions, features, classes)
        values = shap_values[0, :, 1]

    elif shap_values.ndim == 2:
        # Shape:
        # (number of transactions, features)
        values = shap_values[0]

    elif shap_values.ndim == 1:
        values = shap_values

    else:
        raise ValueError(
            f"Unexpected SHAP shape: {shap_values.shape}"
        )

    # Make sure lengths match
    features = list(transaction_df.columns)

    if len(features) != len(values):
        raise ValueError(
            f"Feature count = {len(features)}, "
            f"SHAP value count = {len(values)}"
        )

    # Create explanation table
    explanation = pd.DataFrame({
        "Feature": features,
        "Impact": values
    })

    # Absolute impact
    explanation["Absolute Impact"] = (
        explanation["Impact"].abs()
    )

    # Sort by strongest impact
    explanation = explanation.sort_values(
        "Absolute Impact",
        ascending=False
    )

    return explanation.head(5)


# Select a fraudulent transaction
fraud_transaction = (
    df[df["Class"] == 1]
    .drop("Class", axis=1)
    .iloc[0]
    .to_dict()
)

# Generate explanation
result = explain_transaction(fraud_transaction)

print("-----------------------------------")
print("       RISKGUARD AI EXPLANATION")
print("-----------------------------------")

print("\nTop Risk Factors:")

for _, row in result.iterrows():

    if row["Impact"] > 0:
        direction = "increased"
    else:
        direction = "decreased"

    print(
        f"• {row['Feature']} {direction} "
        f"fraud risk "
        f"(impact: {row['Impact']:.4f})"
    )

print("-----------------------------------")