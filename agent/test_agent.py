from risk_agent import risk_agent


# Example transaction
risk_score = 55
risk_level = "MEDIUM"
amount = 5000
fraud_probability = 0.55


result = risk_agent(
    risk_score,
    risk_level,
    amount,
    fraud_probability
)


print("===================================")
print("        RISKGUARD AI AGENT")
print("===================================")

print(f"Risk Score       : {risk_score}/100")
print(f"Fraud Probability: {fraud_probability * 100:.2f}%")
print(f"Transaction      : ₹{amount}")
print(f"Priority         : {result['priority']}")
print(f"Decision         : {result['decision']}")

print("\nAgent Explanation:")
print(result["explanation"])

print("===================================")