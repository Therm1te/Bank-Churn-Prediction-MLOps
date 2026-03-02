"""
Streamlit Frontend for Bank Churn Prediction
=============================================
Interactive UI that sends customer features to the FastAPI backend
and displays the churn prediction result.
"""

import streamlit as st
import requests
import json

# ---- Page Configuration ----
st.set_page_config(
    page_title="Bank Churn Predictor",
    page_icon="🏦",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---- Custom CSS ----
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #1a5276;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #5d6d7e;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin-top: 1rem;
    }
    .no-churn {
        background-color: #d5f5e3;
        border: 2px solid #27ae60;
    }
    .yes-churn {
        background-color: #fadbd8;
        border: 2px solid #e74c3c;
    }
    .stButton > button {
        width: 100%;
        font-size: 1.2rem;
        font-weight: bold;
        padding: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)

# ---- API Configuration ----
API_URL = "http://127.0.0.1:8000"

# ---- Header ----
st.markdown('<h1 class="main-title">🏦 Bank Churn Prediction</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Predict whether a bank customer will leave using Machine Learning</p>',
    unsafe_allow_html=True,
)

# ---- Check Backend Health ----
try:
    health = requests.get(f"{API_URL}/", timeout=3)
    if health.status_code == 200:
        data = health.json()
        st.sidebar.success(f"✅ API Connected | Model: {data.get('model', 'Unknown')}")
    else:
        st.sidebar.warning("⚠️ API responded with unexpected status")
except requests.exceptions.ConnectionError:
    st.sidebar.error("❌ Backend API is not running! Start it with:\n\n`uvicorn app.main:app --reload`")

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.markdown(
    "This app uses a trained ML model served via FastAPI "
    "to predict bank customer churn. Enter customer details "
    "and click **Predict** to see the result."
)

# ---- Input Form ----
st.markdown("### 📝 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    credit_score = st.number_input(
        "Credit Score",
        min_value=350,
        max_value=850,
        value=650,
        step=1,
        help="Customer's credit score (350-850)",
    )
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=92,
        value=40,
        step=1,
        help="Customer's age",
    )
    tenure = st.number_input(
        "Tenure (years)",
        min_value=0,
        max_value=10,
        value=5,
        step=1,
        help="Number of years with the bank",
    )

with col2:
    geography = st.selectbox(
        "Geography",
        options=["France", "Germany", "Spain"],
        index=0,
        help="Customer's country",
    )
    gender = st.selectbox(
        "Gender",
        options=["Male", "Female"],
        index=0,
        help="Customer's gender",
    )
    balance = st.number_input(
        "Account Balance ($)",
        min_value=0.0,
        max_value=300000.0,
        value=50000.0,
        step=1000.0,
        format="%.2f",
        help="Current account balance",
    )

with col3:
    num_products = st.slider(
        "Number of Products",
        min_value=1,
        max_value=4,
        value=2,
        help="Number of bank products the customer uses",
    )
    has_cr_card = st.selectbox(
        "Has Credit Card?",
        options=[("Yes", 1), ("No", 0)],
        format_func=lambda x: x[0],
        help="Whether the customer has a credit card",
    )
    is_active = st.selectbox(
        "Is Active Member?",
        options=[("No", 0), ("Yes", 1)],
        format_func=lambda x: x[0],
        help="Whether the customer is an active member",
    )

estimated_salary = st.number_input(
    "Estimated Annual Salary ($)",
    min_value=0.0,
    max_value=250000.0,
    value=80000.0,
    step=1000.0,
    format="%.2f",
    help="Customer's estimated annual salary",
)

st.markdown("---")

# ---- Predict Button ----
if st.button("🔮 Predict Churn", type="primary"):
    # Build payload
    payload = {
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": has_cr_card[1],
        "IsActiveMember": is_active[1],
        "EstimatedSalary": estimated_salary,
    }

    with st.spinner("Sending request to API..."):
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=10,
            )

            if response.status_code == 200:
                result = response.json()
                prediction = result["prediction"]
                probability = result["probability"]
                label = result["label"]
                model_used = result.get("model_used", "Unknown")

                # Display result
                if prediction == 0:
                    st.markdown(
                        f"""
                        <div class="result-box no-churn">
                            <h2 style="color: #27ae60; margin:0;">✅ No Churn</h2>
                            <p style="font-size: 1.3rem; margin: 0.5rem 0;">
                                This customer is <strong>unlikely to leave</strong>.
                            </p>
                            <p style="color: #555;">
                                Churn Probability: <strong>{probability:.2%}</strong> | Model: {model_used}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="result-box yes-churn">
                            <h2 style="color: #e74c3c; margin:0;">⚠️ Churn Predicted</h2>
                            <p style="font-size: 1.3rem; margin: 0.5rem 0;">
                                This customer is <strong>likely to leave</strong>!
                            </p>
                            <p style="color: #555;">
                                Churn Probability: <strong>{probability:.2%}</strong> | Model: {model_used}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Show request/response details
                with st.expander("📋 Request & Response Details"):
                    st.markdown("**Request Payload:**")
                    st.json(payload)
                    st.markdown("**API Response:**")
                    st.json(result)

            else:
                st.error(f"API Error ({response.status_code}): {response.text}")

        except requests.exceptions.ConnectionError:
            st.error(
                "❌ Could not connect to the backend API. "
                "Make sure it's running:\n\n"
                "```\ncd backend\nuvicorn app.main:app --reload\n```"
            )
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# ---- Footer ----
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #aaa; font-size: 0.85rem;'>"
    "Bank Churn Prediction App | MLOps Assignment 02 | FastAPI + Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
