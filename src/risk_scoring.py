import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/fraud_model.pkl")

# Load separate scalers
time_scaler = joblib.load("models/time_scaler.pkl")
amount_scaler = joblib.load("models/amount_scaler.pkl")


def calculate_risk(transaction):
    """
    Calculate fraud probability, risk score,
    and risk level for a transaction.
    """

    transaction_df = pd.DataFrame([transaction])

    # Scale Time
    transaction_df["Time"] = time_scaler.transform(
        transaction_df[["Time"]]
    )

    # Scale Amount
    transaction_df["Amount"] = amount_scaler.transform(
        transaction_df[["Amount"]]
    )

    # Get probability of fraud
    fraud_probability = model.predict_proba(transaction_df)[0][1]

    # Convert probability to score from 0 to 100
    risk_score = round(fraud_probability * 100, 2)

    # Determine risk level
    if risk_score <= 30:
        risk_level = "LOW"
    elif risk_score <= 70:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"

    return risk_score, risk_level


# Load dataset
df = pd.read_csv("data/creditcard.csv")

# Select one legitimate transaction
legitimate_transaction = df[df["Class"] == 0].drop(
    "Class", axis=1
).iloc[0].to_dict()

# Select one fraudulent transaction
fraud_transaction = df[df["Class"] == 1].drop(
    "Class", axis=1
).iloc[0].to_dict()


# Calculate risk for legitimate transaction
legit_score, legit_level = calculate_risk(legitimate_transaction)

print("-----------------------------------")
print("       LEGITIMATE TRANSACTION")
print("-----------------------------------")
print(f"Risk Score : {legit_score}/100")
print(f"Risk Level : {legit_level}")


# Calculate risk for fraudulent transaction
fraud_score, fraud_level = calculate_risk(fraud_transaction)

print("\n-----------------------------------")
print("        FRAUD TRANSACTION")
print("-----------------------------------")
print(f"Risk Score : {fraud_score}/100")
print(f"Risk Level : {fraud_level}")
print("-----------------------------------")