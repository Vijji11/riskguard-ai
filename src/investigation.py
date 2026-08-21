import pandas as pd
import joblib


# Load model
model = joblib.load("models/fraud_model.pkl")

# Load scalers
time_scaler = joblib.load("models/time_scaler.pkl")
amount_scaler = joblib.load("models/amount_scaler.pkl")


def investigate_transaction(transaction):

    transaction_df = pd.DataFrame([transaction])

    # Scale Time
    transaction_df["Time"] = time_scaler.transform(
        transaction_df[["Time"]]
    )

    # Scale Amount
    transaction_df["Amount"] = amount_scaler.transform(
        transaction_df[["Amount"]]
    )

    # Fraud probability
    probability = model.predict_proba(
        transaction_df
    )[0][1]

    risk_score = round(probability * 100, 2)

    # Determine risk
    if risk_score <= 30:
        risk_level = "LOW"
        action = "ALLOW"

    elif risk_score <= 70:
        risk_level = "MEDIUM"
        action = "VERIFY"

    else:
        risk_level = "HIGH"
        action = "HOLD"

    # Investigation summary
    if risk_level == "LOW":

        summary = (
            "The transaction has a low predicted fraud "
            "probability. No immediate intervention is required."
        )

    elif risk_level == "MEDIUM":

        summary = (
            "The transaction shows some characteristics "
            "associated with fraudulent activity. "
            "Additional verification is recommended."
        )

    else:

        summary = (
            "The transaction has a high predicted fraud "
            "probability. It should be held temporarily "
            "and investigated before completion."
        )

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "action": action,
        "summary": summary
    }