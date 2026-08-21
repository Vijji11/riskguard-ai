import streamlit as st
import pandas as pd
import joblib
import sys
import shap
import numpy as np


# ============================================================
# PROJECT IMPORTS
# ============================================================

sys.path.append("src")
sys.path.append("agent")

from investigation import investigate_transaction
from risk_agent import risk_agent


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(
    "models/fraud_model.pkl"
)

time_scaler = joblib.load(
    "models/time_scaler.pkl"
)

amount_scaler = joblib.load(
    "models/amount_scaler.pkl"
)


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(
    "data/creditcard.csv"
)


# ============================================================
# SHAP EXPLAINER
# ============================================================

explainer = shap.TreeExplainer(model)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RiskGuard AI",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("🛡️ RiskGuard AI")

st.subheader(
    "AI-Powered Financial Risk Manager"
)

st.write(
    "Analyze financial transactions, detect potential "
    "fraud, calculate risk scores, investigate risk "
    "factors, and receive recommended actions."
)

st.divider()


# ============================================================
# DATASET STATISTICS
# ============================================================

total_transactions = len(df)

fraud_transactions = int(
    df["Class"].sum()
)

legitimate_transactions = (
    total_transactions -
    fraud_transactions
)

fraud_rate = (
    fraud_transactions /
    total_transactions
) * 100


# ============================================================
# TOP METRICS
# ============================================================

st.header("📊 Risk Overview")

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Total Transactions",
        f"{total_transactions:,}"
    )


with col2:

    st.metric(
        "Fraudulent",
        f"{fraud_transactions:,}"
    )


with col3:

    st.metric(
        "Legitimate",
        f"{legitimate_transactions:,}"
    )


with col4:

    st.metric(
        "Fraud Rate",
        f"{fraud_rate:.4f}%"
    )


st.divider()


# ============================================================
# ANALYTICS
# ============================================================

st.header("📈 Transaction Analytics")


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
        "Fraud Detection Statistics"
    )

    st.write(
        f"**Total:** {total_transactions:,}"
    )

    st.write(
        f"**Fraud:** {fraud_transactions:,}"
    )

    st.write(
        f"**Legitimate:** "
        f"{legitimate_transactions:,}"
    )

    st.write(
        f"**Fraud Rate:** "
        f"{fraud_rate:.4f}%"
    )

    st.info(
        "Fraud detection is challenging because "
        "fraudulent transactions are extremely rare "
        "compared with legitimate transactions."
    )


st.divider()


# ============================================================
# TRANSACTION ANALYSIS
# ============================================================

st.header("🔍 Transaction Analysis")


selection_type = st.radio(
    "Choose transaction selection method:",
    [
        "Transaction Number",
        "Find Fraud Transaction",
        "Find Legitimate Transaction"
    ],
    horizontal=True
)


# ============================================================
# TRANSACTION SELECTION
# ============================================================

if selection_type == "Transaction Number":

    selected_index = st.number_input(
        "Enter Transaction Number",
        min_value=0,
        max_value=len(df) - 1,
        value=0,
        step=1
    )

    selected_index = int(
        selected_index
    )


elif selection_type == "Find Fraud Transaction":

    fraud_indices = (
        df[df["Class"] == 1]
        .index
        .tolist()
    )

    selected_index = st.selectbox(
        "Select Fraud Transaction",
        fraud_indices
    )


else:

    legitimate_indices = (
        df[df["Class"] == 0]
        .index
        .tolist()
    )

    selected_index = st.selectbox(
        "Select Legitimate Transaction",
        legitimate_indices
    )


selected_transaction = df.loc[
    selected_index
]


# ============================================================
# TRANSACTION DETAILS
# ============================================================

st.subheader(
    "Transaction Details"
)


col1, col2, col3 = st.columns(3)


with col1:

    st.write(
        f"**Transaction:** "
        f"{selected_index}"
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

analyze = st.button(
    "🔍 Analyze Transaction",
    use_container_width=True
)


if analyze:

    # ========================================================
    # CREATE TRANSACTION
    # ========================================================

    transaction = (
        selected_transaction
        .drop("Class")
        .to_dict()
    )


    # ========================================================
    # PREPARE DATA
    # ========================================================

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


    # ========================================================
    # MACHINE LEARNING PREDICTION
    # ========================================================

    fraud_probability = (
        model.predict_proba(
            transaction_df
        )[0][1]
    )


    risk_score = round(
        fraud_probability * 100,
        2
    )


    # ========================================================
    # RISK CLASSIFICATION
    # ========================================================

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
    # RISK BAR
    # ========================================================

    st.subheader(
        "Risk Level"
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
    # AI INVESTIGATION
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
    # RISKGUARD AI AGENT
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
    # SHAP EXPLANATION
    # ========================================================

    st.divider()

    st.subheader(
        "🔎 Why did RiskGuard make this decision?"
    )


    st.write(
        "These model features had the strongest "
        "influence on the prediction:"
    )


    # Calculate SHAP values

    shap_values = explainer.shap_values(
        transaction_df
    )


    shap_values = np.asarray(
        shap_values
    )


    # Handle SHAP versions

    if shap_values.ndim == 3:

        feature_impacts = (
            shap_values[0, :, 1]
        )

    elif shap_values.ndim == 2:

        feature_impacts = (
            shap_values[0]
        )

    elif shap_values.ndim == 1:

        feature_impacts = (
            shap_values
        )

    else:

        st.error(
            "Unexpected SHAP output format."
        )

        feature_impacts = np.zeros(
            len(transaction_df.columns)
        )


    # Make sure lengths match

    if len(feature_impacts) == len(
        transaction_df.columns
    ):

        explanation_df = pd.DataFrame({

            "Feature":
                transaction_df.columns,

            "Impact":
                feature_impacts

        })


        explanation_df[
            "Absolute Impact"
        ] = (
            explanation_df[
                "Impact"
            ].abs()
        )


        top_features = (
            explanation_df
            .sort_values(
                "Absolute Impact",
                ascending=False
            )
            .head(5)
        )


        for _, row in (
            top_features.iterrows()
        ):

            if row["Impact"] > 0:

                st.write(
                    f"🔴 **{row['Feature']}** "
                    f"increased fraud risk "
                    f"({row['Impact']:.4f})"
                )

            else:

                st.write(
                    f"🟢 **{row['Feature']}** "
                    f"reduced fraud risk "
                    f"({row['Impact']:.4f})"
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
    # RISK INTERPRETATION
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