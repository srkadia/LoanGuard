# 💼 LoanGuard: AI-Powered Loan Default Prediction

LoanGuard is an end-to-end **machine learning system** that predicts the likelihood of loan default using real-world data from Lending Club. The application is designed for scalability, interpretability, and real-time usage—making it suitable for both demos and production environments.

Built with **FastAPI, Docker, MLflow**, and a **Next.js** frontend, the system is deployable on cloud platforms like **AWS EC2**, and follows best practices in MLOps and ML engineering.

---

## 🚀 Project Highlights

- 🔍 **Loan Default Prediction**: Real-time risk prediction using loan application inputs.
- 🧠 **ML Models**: Neural Networks, XGBoost, Random Forest, Logistic Regression.
- 🧪 **MLOps Integration**: Full lifecycle tracking via MLflow, hyperparameter tuning with Optuna.
- 🧱 **Modular ML Pipelines**: Clean, reusable code structure for preprocessing, training, and inference.
- ⚡ **REST API**: FastAPI-powered backend for quick inference and model versioning.
- 🖥️ **Interactive UI**: Built with Next.js for seamless user interaction.
- 📦 **Dockerized Deployment(Dev Branch)**: Easy containerization for local or cloud use.

---

## 📊 Dataset Overview

- **Source**: Lending Club Loan Dataset (Kaggle/UCI)
- **Target Variable**: `loan_status` (binary: Default or Not Default)
- **Features Used**:
  - Loan amount, interest rate, term, purpose
  - Employment length, annual income, DTI
  - Credit history length, delinquency, open accounts, revolving balance

---

## 🔬 Data Preprocessing & Feature Engineering

- Missing value imputation using median/mode strategies
- Feature scaling via StandardScaler / MinMaxScaler
- Categorical encoding using One-Hot and Ordinal Encoders
- Derived features: credit-to-income ratio, loan-to-income ratio, etc.
- Train/Validation/Test split for robust evaluation

---

## 🧠 Machine Learning Models

The system supports multiple models for experimentation and benchmarking:

- ✅ Neural Network Classifier (built with PyTorch/Sklearn)
- 🌲 Random Forest
- 📈 Logistic Regression
- ⚡ XGBoost Classifier

**Optimization**:
- Hyperparameter tuning with **Optuna**
- Experiment tracking with **MLflow**
- Model versioning and artifact logging

---

## ⚙️ API Endpoints (FastAPI)

| Endpoint          | Method | Description |
|-------------------|--------|-------------|
| `/predict`        | POST   | Returns default risk prediction from input JSON |
---

## 🖥️ Frontend (Next.js)

- Clean dashboard to input loan application data
- Real-time response from FastAPI backend
- Displays prediction and model confidence score
- UI built with modern component libraries and hooks

---

## 🚢 Deployment

- **Containerized** using Docker and Docker Compose
- **Cloud Deployment** on AWS EC2 (tested), GCP, or Heroku
- Optional use of Nginx for API + UI reverse proxy
- MLOps support with MLflow Tracking Server and local SQLite/remote S3 backend

---

## 🛠 Tech Stack

- **Backend**: Python, FastAPI, MLflow, Optuna
- **ML**: Scikit-learn, Tensorflow, PyTorch
- **Frontend**: Next.js, Axios, Tailwind CSS
- **Deployment**: Docker (dev branch), AWS EC2 (tested)

---

## 🤝 Contributing

Contributions are welcome! If you’d like to improve the project—whether it's new models, UI improvements, or backend optimization—feel free to open a pull request or create an issue.

---

## 📬 Contact

- **Email**: srkkadia@gmail.com  
- **LinkedIn**: [srkadia](https://linkedin.com/in/srkadia)  
- **GitHub**: [srkadia](https://github.com/srkadia)

---

## 📎 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

