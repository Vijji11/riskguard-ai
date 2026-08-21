import pandas as pd

from investigation import investigate_transaction


# Load dataset
df = pd.read_csv("data/creditcard.csv")


# Get first fraud transaction
transaction = (
    df[df["Class"] == 1]
    .drop("Class", axis=1)
    .iloc[0]
    .to_dict()
)


# Investigate
result = investigate_transaction(transaction)


print("===================================")
print("       AI RISK INVESTIGATION")
print("===================================")

print(f"Risk Score : {result['risk_score']}/100")
print(f"Risk Level : {result['risk_level']}")
print(f"Action     : {result['action']}")

print("\nInvestigation:")
print(result["summary"])

print("===================================")