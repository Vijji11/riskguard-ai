import streamlit as st
import pandas as pd
import joblib
import sys

# -----------------------------
# Import project modules
# -----------------------------

sys.path.append("src")
sys.path.append("agent")

from investigation import investigate_transaction
from risk_agent import risk_agent


# -----------------------------
# Load model and scalers
# -----------------------------

model = joblib.load("models/fraud_model.pkl")

time_scaler = joblib.load(
    "models/time_scaler.pkl"
)

amount_scaler = joblib.load(
    "models/amount_scaler.pkl"
)


# -----------------------------
# Load dataset
# -----------------------------

df = pd.read_csv(
    "data/creditcard.csv"
)


# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="RiskGuard AI",
    page_icon="🛡️",
    layout="wide"
)


# -----------------------------
# Header
# -----------------------------

st.title("🛡️ RiskGuard AI")

st.subheader(
    "AI-Powered Financial Risk Manager"
)

st.write(
    "Analyze financial transactions and identify "
    "potential fraud risk using machine learning, "
    "risk analysis, and an AI decision engine."
)

st.divider()


# ============================================================
# DASHBOARD METRICS
# ============================================================

total_transactions = len(df)

fraud_transactions = int(
    df["Class"].sum()
)

legitimate_transactions = (
    total_transactions - fraud_transactions
)

fraud_rate = (
    fraud_transactions / total_transactions
) * 100


col1, col2, col3, col4 = st.columns(4)


with col1:
    st.metric(
        "Total Transactions",
        f"{total_transactions:,}"
    )


with col2:
    st.metric(
        "Fraudulent Transactions",
        f"{fraud_transactions:,}"
    )


with col3:
    st.metric(
        "Legitimate Transactions",
        f"{legitimate_transactions:,}"
    )


with col4:
    st.metric(
        "Fraud Rate",
        f"{fraud_rate:.4f}%"
    )


st.divider()


# ============================================================
# RISK ANALYTICS
# ============================================================

st.header("📊 Risk Analytics")


col1, col2 = st.columns(2)


with col1:

    st.subheader(
        "Transaction Distribution"
    )

    chart_data = pd.DataFrame({
        "Type": [
            "Legitimate",
            "Fraudulent"
        ],
        "Count": [
            legitimate_transactions,
            fraud_transactions
        ]
    })

    st.bar_chart(
        chart_data.set_index("Type")
    )


with col2:

    st.subheader(
        "Fraud Detection Overview"
    )

    st.write(
        f"**Total transactions:** "
        f"{total_transactions:,}"
    )

    st.write(
        f"**Fraudulent transactions:** "
        f"{fraud_transactions:,}"
    )

    st.write(
        f"**Fraud rate:** "
        f"{fraud_rate:.4f}%"
    )

    st.info(
        "The dataset is highly imbalanced because "
        "fraudulent transactions represent only a "
        "small percentage of all transactions."
    )


st.divider()


# ============================================================
# TRANSACTION ANALYSIS
# ============================================================

st.header("🔍 Transaction Analysis")


transaction_number = st.number_input(
    "Enter Transaction Number",
    min_value=0,
    max_value=len(df) - 1,
    value=0,
    step=1
)


selected_transaction = df.iloc[
    transaction_number
]


# ============================================================
# TRANSACTION DETAILS
# ============================================================

st.write("### Transaction Details")


col1, col2, col3 = st.columns(3)


with col1:

    st.write(
        f"**Transaction Number:** "
        f"{transaction_number}"
    )


with col2:

    st.write(
        f"**Time:** "
        f"{selected_transaction['Time']:.2f}"
    )


with col3:

    st.write(
        f"**Amount:** "
        f"${selected_transaction['Amount']:.2f}"
    )


# ============================================================
# ANALYZE BUTTON
# ============================================================

if st.button(
    "🔍 Analyze Transaction",
    use_container_width=True
):

    # --------------------------------------------------------
    # Create transaction dictionary
    # --------------------------------------------------------

    transaction = (
        selected_transaction
        .drop("Class")
        .to_dict()
    )


    # --------------------------------------------------------
    # Prepare transaction
    # --------------------------------------------------------

    transaction_df = pd.DataFrame(
        [transaction]
    )


    # Scale Time

    transaction_df["Time"] = (
        time_scaler.transform(
            transaction_df[["Time"]]
        )
    )


    # Scale Amount

    transaction_df["Amount"] = (
        amount_scaler.transform(
            transaction_df[["Amount"]]
        )
    )


    # --------------------------------------------------------
    # ML Prediction
    # --------------------------------------------------------

    fraud_probability = (
        model.predict_proba(
            transaction_df
        )[0][1]
    )


    risk_score = round(
        fraud_probability * 100,
        2
    )


    # --------------------------------------------------------
    # Risk Level
    # --------------------------------------------------------

    if risk_score <= 30:

        risk_level = "LOW"

        action = "ALLOW"

        message = (
            "Transaction appears safe."
        )

    elif risk_score <= 70:

        risk_level = "MEDIUM"

        action = "VERIFY"

        message = (
            "Additional verification "
            "is recommended."
        )

    else:

        risk_level = "HIGH"

        action = "HOLD"

        message = (
            "Transaction should be temporarily "
            "held for investigation."
        )


    # ========================================================
    # RISK ASSESSMENT
    # ========================================================

    st.divider()

    st.header(
        "🛡️ Risk Assessment"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Risk Score",
            f"{risk_score}/100"
        )


    with col2:

        st.metric(
            "Risk Level",
            risk_level
        )


    with col3:

        st.metric(
            "Fraud Probability",
            f"{fraud_probability * 100:.2f}%"
        )


    # ========================================================
    # RISK SCORE VISUALIZATION
    # ========================================================

    st.subheader(
        "Risk Score Visualization"
    )

    st.progress(
        min(
            risk_score / 100,
            1.0
        )
    )


    # ========================================================
    # RECOMMENDED ACTION
    # ========================================================

    st.subheader(
        "Recommended Action"
    )


    if action == "ALLOW":

        st.success(
            f"✅ {action} — {message}"
        )


    elif action == "VERIFY":

        st.warning(
            f"⚠️ {action} — {message}"
        )


    else:

        st.error(
            f"🚨 {action} — {message}"
        )


    # ========================================================
    # AI RISK INVESTIGATION
    # ========================================================

    st.divider()

    st.subheader(
        "🤖 AI Risk Investigation"
    )


    investigation = (
        investigate_transaction(
            transaction
        )
    )


    st.write(
        investigation["summary"]
    )


    col1, col2 = st.columns(2)


    with col1:

        st.write(
            f"**Risk Level:** "
            f"{investigation['risk_level']}"
        )


    with col2:

        st.write(
            f"**Recommended Action:** "
            f"{investigation['action']}"
        )


    # ========================================================
    # AI RISK AGENT
    # ========================================================

    st.divider()

    st.subheader(
        "🧠 RiskGuard AI Agent"
    )


    agent_result = risk_agent(

        risk_score=risk_score,

        risk_level=risk_level,

        amount=selected_transaction[
            "Amount"
        ],

        fraud_probability=fraud_probability
    )


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Agent Priority",
            agent_result["priority"]
        )


    with col2:

        st.metric(
            "Agent Decision",
            agent_result["decision"]
        )


    st.info(
        agent_result["explanation"]
    )


    # ========================================================
    # TRANSACTION FEATURES
    # ========================================================

    st.divider()

    st.subheader(
        "📋 Transaction Features"
    )


    feature_data = (
        selected_transaction
        .drop("Class")
        .to_frame(
            name="Value"
        )
    )


    st.dataframe(
        feature_data,
        use_container_width=True
    )


    # ========================================================
    # DATASET VERIFICATION
    # ========================================================

    st.divider()

    st.subheader(
        "📊 Dataset Verification"
    )


    actual_class = (
        selected_transaction["Class"]
    )


    if actual_class == 1:

        st.error(
            "⚠️ This transaction is labeled "
            "as FRAUD in the dataset."
        )

    else:

        st.success(
            "✅ This transaction is labeled "
            "as LEGITIMATE in the dataset."
        )


    # ========================================================
    # MODEL INTERPRETATION
    # ========================================================

    st.divider()

    st.subheader(
        "ℹ️ Risk Interpretation"
    )


    if risk_level == "LOW":

        st.write(
            "The model predicts a relatively low "
            "probability of fraudulent activity. "
            "The recommended action is to allow "
            "the transaction."
        )


    elif risk_level == "MEDIUM":

        st.write(
            "The model identifies moderate risk. "
            "The recommended action is additional "
            "verification before completing the "
            "transaction."
        )


    else:

        st.write(
            "The model identifies high risk. "
            "The recommended action is to temporarily "
            "hold the transaction for investigation."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "RiskGuard AI | AI-Powered Financial Risk Manager"
)