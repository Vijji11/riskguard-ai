import pandas as pd
import joblib

from risk_decision import get_recommended_action

# Load trained model
model = joblib.load("models/fraud_model.pkl")

# Load scalers
time_scaler = joblib.load("models/time_scaler.pkl")
amount_scaler = joblib.load("models/amount_scaler.pkl")


def analyze_transaction(transaction):
    """
    Analyze a transaction and return:
    fraud probability, risk score, risk level,
    and recommended action.
    """

    # Convert transaction to DataFrame
    transaction_df = pd.DataFrame([transaction])

    # Scale Time
    transaction_df["Time"] = time_scaler.transform(
        transaction_df[["Time"]]
    )

    # Scale Amount
    transaction_df["Amount"] = amount_scaler.transform(
        transaction_df[["Amount"]]
    )

    # Predict fraud probability
    fraud_probability = model.predict_proba(
        transaction_df
    )[0][1]

    # Convert probability to score
    risk_score = round(fraud_probability * 100, 2)

    # Get risk decision
    decision = get_recommended_action(risk_score)

    return {
        "fraud_probability": round(fraud_probability, 4),
        "risk_score": risk_score,
        "risk_level": decision["risk_level"],
        "action": decision["action"],
        "message": decision["message"]
    }


# Load dataset
df = pd.read_csv("data/creditcard.csv")

# Get one legitimate transaction
legitimate_transaction = (
    df[df["Class"] == 0]
    .drop("Class", axis=1)
    .iloc[0]
    .to_dict()
)

# Get one fraud transaction
fraud_transaction = (
    df[df["Class"] == 1]
    .drop("Class", axis=1)
    .iloc[0]
    .to_dict()
)


# Analyze legitimate transaction
legit_result = analyze_transaction(
    legitimate_transaction
)

print("\n===================================")
print("     LEGITIMATE TRANSACTION")
print("===================================")
print(f"Fraud Probability : {legit_result['fraud_probability']}")
print(f"Risk Score        : {legit_result['risk_score']}/100")
print(f"Risk Level        : {legit_result['risk_level']}")
print(f"Action            : {legit_result['action']}")
print(f"Message           : {legit_result['message']}")


# Analyze fraud transaction
fraud_result = analyze_transaction(
    fraud_transaction
)

print("\n===================================")
print("        FRAUD TRANSACTION")
print("===================================")
print(f"Fraud Probability : {fraud_result['fraud_probability']}")
print(f"Risk Score        : {fraud_result['risk_score']}/100")
print(f"Risk Level        : {fraud_result['risk_level']}")
print(f"Action            : {fraud_result['action']}")
print(f"Message           : {fraud_result['message']}")
print("===================================")