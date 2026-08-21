def risk_agent(risk_score, risk_level, amount, fraud_probability):
    """
    AI Risk Manager Agent.
    Analyzes risk information and recommends an action.
    """

    if risk_level == "LOW":

        decision = "ALLOW"

        explanation = (
            "The transaction has a low predicted fraud risk. "
            "No immediate intervention is required."
        )

        priority = "LOW"

    elif risk_level == "MEDIUM":

        decision = "VERIFY"

        explanation = (
            "The transaction has moderate fraud risk. "
            "Additional customer verification is recommended "
            "before completing the transaction."
        )

        priority = "MEDIUM"

    else:

        decision = "HOLD"

        explanation = (
            "The transaction has high predicted fraud risk. "
            "Temporarily hold the transaction and investigate "
            "before allowing it to proceed."
        )

        priority = "HIGH"

    return {
        "decision": decision,
        "priority": priority,
        "explanation": explanation
    }