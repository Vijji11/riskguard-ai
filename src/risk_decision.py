def get_recommended_action(risk_score):
    """
    Decide what action should be taken
    based on the transaction risk score.
    """

    if risk_score <= 30:
        return {
            "risk_level": "LOW",
            "action": "ALLOW",
            "message": "Transaction appears safe."
        }

    elif risk_score <= 70:
        return {
            "risk_level": "MEDIUM",
            "action": "VERIFY",
            "message": "Additional verification is recommended."
        }

    else:
        return {
            "risk_level": "HIGH",
            "action": "HOLD",
            "message": "Transaction should be temporarily held for investigation."
        }


# Test the decision engine
test_scores = [10, 55, 90]

for score in test_scores:

    result = get_recommended_action(score)

    print("-----------------------------------")
    print(f"Risk Score : {score}/100")
    print(f"Risk Level : {result['risk_level']}")
    print(f"Action     : {result['action']}")
    print(f"Message    : {result['message']}")