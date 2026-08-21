import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    accuracy_score
)
from sklearn.model_selection import train_test_split


# -----------------------------
# Load dataset and model
# -----------------------------

df = pd.read_csv("data/creditcard.csv")

model = joblib.load(
    "models/fraud_model.pkl"
)

time_scaler = joblib.load(
    "models/time_scaler.pkl"
)

amount_scaler = joblib.load(
    "models/amount_scaler.pkl"
)


# -----------------------------
# Prepare data
# -----------------------------

X = df.drop("Class", axis=1)
y = df["Class"]

X["Time"] = time_scaler.transform(
    X[["Time"]]
)

X["Amount"] = amount_scaler.transform(
    X[["Amount"]]
)


# -----------------------------
# Train/test split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------
# Predictions
# -----------------------------

y_pred = model.predict(X_test)

y_probability = model.predict_proba(
    X_test
)[:, 1]


accuracy = accuracy_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

matrix = confusion_matrix(
    y_test,
    y_pred
)


# -----------------------------
# Page
# -----------------------------

st.title("📈 RiskGuard AI — Model Performance")

st.write(
    "Performance evaluation of the fraud detection "
    "machine learning model."
)

st.divider()


# -----------------------------
# Metrics
# -----------------------------

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Accuracy",
        f"{accuracy * 100:.2f}%"
    )


with col2:

    st.metric(
        "ROC-AUC",
        f"{roc_auc:.4f}"
    )


with col3:

    st.metric(
        "Fraud Precision",
        f"{report['1']['precision'] * 100:.2f}%"
    )


with col4:

    st.metric(
        "Fraud Recall",
        f"{report['1']['recall'] * 100:.2f}%"
    )


st.divider()


# -----------------------------
# Confusion Matrix
# -----------------------------

st.subheader("Confusion Matrix")

matrix_df = pd.DataFrame(
    matrix,
    index=[
        "Actual Legitimate",
        "Actual Fraud"
    ],
    columns=[
        "Predicted Legitimate",
        "Predicted Fraud"
    ]
)

st.dataframe(
    matrix_df,
    use_container_width=True
)


st.divider()


# -----------------------------
# Classification Report
# -----------------------------

st.subheader(
    "Classification Report"
)

report_df = pd.DataFrame(
    report
).transpose()

st.dataframe(
    report_df,
    use_container_width=True
)


st.divider()


st.info(
    "The model is evaluated using fraud-focused metrics "
    "such as precision, recall, F1-score and ROC-AUC "
    "because fraud detection is a highly imbalanced "
    "classification problem."
)
