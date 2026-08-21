# 🛡️ RiskGuard AI

## AI-Powered Financial Risk Manager

RiskGuard AI is an intelligent financial risk management system that analyzes financial transactions, predicts potential fraud, calculates a risk score, and recommends appropriate actions.

The project combines machine learning, explainable AI, risk decision logic, and an AI risk agent into a single financial risk management platform.

---

## 🎯 Problem Statement

Financial institutions and payment platforms process large numbers of transactions every day. Detecting fraudulent transactions quickly is challenging because fraudulent transactions are rare and can be difficult to distinguish from legitimate activity.

RiskGuard AI aims to identify potentially fraudulent transactions and provide an understandable risk assessment to support faster decision-making.

---

## 💡 Solution

RiskGuard AI analyzes a transaction using a machine learning fraud detection model.

The system:

1. Processes transaction data.
2. Predicts the probability of fraud.
3. Converts the probability into a risk score.
4. Classifies the transaction as Low, Medium, or High risk.
5. Recommends an action.
6. Generates a risk investigation summary.
7. Uses an AI risk decision agent to recommend the next step.
8. Displays the results through an interactive dashboard.

---

## 🏗️ System Architecture

```text
Transaction
     ↓
Data Preprocessing
     ↓
Random Forest Model
     ↓
Fraud Probability
     ↓
Risk Score
     ↓
Risk Level
     ↓
Risk Investigation
     ↓
AI Risk Agent
     ↓
Recommended Action
     ↓
Streamlit Dashboard