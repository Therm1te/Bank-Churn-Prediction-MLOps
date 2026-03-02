# 🏦 Bank Churn Prediction — MLOps

A machine-learning application that predicts whether a bank customer will churn (leave), served through a **FastAPI** REST API and consumed via an interactive **Streamlit** frontend.

> **MLOps Assignment 02** · FastAPI + Streamlit · XGBoost / Random Forest / Logistic Regression

---

## 📁 Project Structure

```
Bank-Churn-Prediction-MLOps/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py            # FastAPI app — routes & validation
│   │   └── model.py           # Model loading & prediction logic
│   ├── model.pkl              # Trained ML model artifact
│   └── requirements.txt       # Backend dependencies
├── frontend/
│   ├── app.py                 # Streamlit UI for churn prediction
│   └── requirements.txt       # Frontend dependencies
├── data/
│   └── bank_churn_modelling.csv   # Dataset
├── .gitignore
├── LICENSE                    # MIT License
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Python      | 3.10+   |
| pip         | latest  |

### 1. Clone the Repository

```bash
git clone https://github.com/Therm1te/Bank-Churn-Prediction-MLOps.git
cd Bank-Churn-Prediction-MLOps
```

### 2. Create & Activate Virtual Environment

```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Backend
pip install -r backend/requirements.txt

# Frontend
pip install -r frontend/requirements.txt
```

### 4. Run the Backend (FastAPI)

```bash
cd backend
uvicorn app.main:app --reload
```

The API will be live at **<http://127.0.0.1:8000>**

| Endpoint       | Method | Description                        |
|----------------|--------|------------------------------------|
| `/`            | GET    | Welcome message & health check     |
| `/predict`     | POST   | Predict customer churn             |
| `/docs`        | GET    | Interactive Swagger documentation  |

### 5. Run the Frontend (Streamlit)

Open a **new terminal**, activate the virtual environment, then:

```bash
cd frontend
streamlit run app.py
```

The UI will open at **<http://localhost:8501>**

---

## 📊 API Reference

### `POST /predict`

**Request Body (JSON):**

```json
{
  "CreditScore": 650,
  "Geography": "France",
  "Gender": "Male",
  "Age": 40,
  "Tenure": 5,
  "Balance": 50000.0,
  "NumOfProducts": 2,
  "HasCrCard": 1,
  "IsActiveMember": 0,
  "EstimatedSalary": 80000.0
}
```

**Response:**

```json
{
  "prediction": 0,
  "probability": 0.12,
  "label": "No",
  "model_used": "XGBoost"
}
```

### cURL Example

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "CreditScore": 650,
    "Geography": "France",
    "Gender": "Male",
    "Age": 40,
    "Tenure": 5,
    "Balance": 50000.0,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 0,
    "EstimatedSalary": 80000.0
  }'
```

---

## 🧠 Model Details

### Input Features

| Feature          | Type    | Description                          |
|------------------|---------|--------------------------------------|
| CreditScore      | `int`   | Customer's credit score (350 – 850)  |
| Geography        | `str`   | Country — France, Germany, or Spain  |
| Gender           | `str`   | Male or Female                       |
| Age              | `int`   | Customer's age (18 – 92)             |
| Tenure           | `int`   | Years with the bank (0 – 10)         |
| Balance          | `float` | Account balance                      |
| NumOfProducts    | `int`   | Number of bank products (1 – 4)      |
| HasCrCard        | `int`   | Has credit card (0 / 1)              |
| IsActiveMember   | `int`   | Is active member (0 / 1)             |
| EstimatedSalary  | `float` | Estimated annual salary              |

### Preprocessing

- **Dropped columns:** RowNumber, CustomerId, Surname (non-predictive)
- **Categorical encoding:** One-Hot Encoding for `Geography` and `Gender` (`drop_first=True`)
  - France → baseline for Geography
  - Female → baseline for Gender

### Models Evaluated

| # | Model               | Notes                  |
|---|---------------------|------------------------|
| 1 | Logistic Regression | Simple baseline        |
| 2 | Random Forest       | Ensemble method        |
| 3 | XGBoost             | Gradient boosting      |

The best-performing model is automatically selected, tuned, and saved to `backend/model.pkl`.

### Evaluation Metrics

- Accuracy · F1 Score · ROC-AUC
- Confusion Matrix · Classification Report (Precision, Recall)

---

## 🏗️ Architecture

```
┌─────────────────┐         ┌──────────────────┐         ┌──────────────┐
│   Streamlit UI  │──POST──▸│   FastAPI API     │──load──▸│  model.pkl   │
│   (Frontend)    │ /predict│   (Backend)       │         │  (ML Model)  │
│  localhost:8501 │◂─JSON───│  localhost:8000   │         └──────────────┘
└─────────────────┘         └──────────────────┘
```

- **Frontend** is fully decoupled from the backend — communicates only via HTTP.
- **API** serves as middleware between the UI and the ML model.
- **Model** can be retrained and swapped without touching the frontend.
- Any HTTP client (web, mobile, CLI) can consume the same API.

---

## �️ Tech Stack

| Layer    | Technology                           |
|----------|--------------------------------------|
| Backend  | FastAPI, Uvicorn, Pydantic           |
| Frontend | Streamlit                            |
| ML       | scikit-learn, XGBoost, pandas, NumPy |
| Language | Python 3.10+                         |

---

## 📝 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
