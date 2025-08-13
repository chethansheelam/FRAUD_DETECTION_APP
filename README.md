# Proactive Fraud Detection: A Machine Learning Approach

This project focuses on developing a high-performance machine learning model to detect fraudulent financial transactions. An interactive web application was built using Streamlit to provide real-time predictions, demonstrating a complete end-to-end data science workflow from data analysis to a deployed user interface.

---

### 🚀 Live Demo

You can view and interact with the live deployed application here:

**https://fraud-detection-app-4s7z.onrender.com**

---

### 📸 App Screenshot


**![Screenshot_13-8-2025_14331_fraud-detection-app-4s7z onrender com](https://github.com/user-attachments/assets/81a4896d-fa3e-4830-b851-32349ea1b53b)


---

### ## 🎯 Project Goal

The objective of this project was to analyze a large transactional dataset and build a predictive model capable of identifying fraudulent transactions with high accuracy, focusing on minimizing financial losses by catching as many fraudulent cases as possible (high recall).

---

### ## 🛠️ Methodology & Workflow

The model was developed following a standard data science pipeline:

1.  **Data Cleaning & EDA**: The initial dataset of over 6 million transactions was explored. The most critical finding was that fraud only occurs in `TRANSFER` and `CASH_OUT` transaction types, which allowed for a more focused modeling approach.
2.  **Feature Engineering**: A new feature, `errorBalance`, was created to capture mathematical discrepancies in account balances, which proved to be a powerful predictor.
3.  **Handling Imbalance**: The dataset was severely imbalanced (only 0.13% fraud cases). This was addressed by applying the **SMOTE (Synthetic Minority Over-sampling Technique)** to the training data.
4.  **Model Training**: An **XGBoost Classifier** was trained on the preprocessed and balanced data.
5.  **Model Evaluation**: The model was evaluated on an unseen test set using metrics appropriate for imbalanced classification.

---

### ## 📊 Results

The final model achieved excellent performance, prioritizing the detection of fraud cases:

* **Recall**: **99%** (The model successfully identified 99% of all actual fraudulent transactions).
* **Precision**: **80%** (When the model predicted fraud, it was correct 80% of the time).
* **Overall Accuracy**: 99.9%

---

### ## 💻 Technologies Used

* **Programming Language**: Python 3
* **Libraries**: Pandas, NumPy, Scikit-learn, XGBoost, Imblearn, Joblib, Streamlit
* **Environment**: Jupyter Notebook, VS Code

---

### ## ⚙️ How to Run This Project Locally

To run this application on your local machine, please follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **(Optional but recommended) Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

The application will then be available in your web browser at `http://localhost:8501`.

### ## 📊 Dataset & Data Dictionary

The dataset for this project is a synthetic log of financial transactions from a mobile money simulator. The data dictionary below describes each column:

step - maps a unit of time in the real world. In this case 1 step is 1 hour of time.

type - CASH-IN, CASH-OUT, DEBIT, PAYMENT and TRANSFER.

amount - amount of the transaction in local currency.

nameOrig - customer who started the transaction.

oldbalanceOrg - initial balance before the transaction.

newbalanceOrig - new balance after the transaction.

nameDest - customer who is the recipient of the transaction.

oldbalanceDest - initial balance of the recipient before the transaction.

newbalanceDest - new balance of the recipient after the transaction.

isFraud - transactions made by fraudulent agents.

isFlaggedFraud - transactions flagged as illegal attempts by the system (a transfer over 200,000 in a single transaction).
