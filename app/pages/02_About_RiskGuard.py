import streamlit as st

st.title("🛡️ About RiskGuard AI")

st.subheader(
    "AI-Powered Financial Risk Manager"
)

st.write("""
RiskGuard AI is an intelligent financial risk management
system designed to identify potentially fraudulent
transactions and support faster financial decision-making.
""")

st.divider()

st.header("🎯 Objective")

st.write("""
The goal of RiskGuard AI is to analyze financial
transactions, estimate fraud probability, generate a
risk score, classify risk levels, and recommend
appropriate actions.
""")

st.header("🤖 How It Works")

st.write("""
1. Transaction data is received.
2. The machine learning model predicts fraud probability.
3. The probability is converted into a risk score.
4. The system determines the risk level.
5. A recommended action is generated.
6. The risk investigation engine provides an explanation.
7. The RiskGuard AI Agent provides a final decision.
""")

st.header("🧠 Technologies")

st.write("""
- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest
- SHAP
- Streamlit
- Joblib
""")

st.header("🚦 Risk Levels")

st.write("""
LOW → Allow transaction

MEDIUM → Request additional verification

HIGH → Temporarily hold transaction for investigation
""")

st.divider()

st.caption(
    "RiskGuard AI — AI Risk Manager"
)