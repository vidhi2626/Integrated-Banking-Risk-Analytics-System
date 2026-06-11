import pickle
import numpy as np

loan_model = pickle.load(open("model/loan_model.pkl", "rb"))
fraud_model = pickle.load(open("model/fraud_model.pkl", "rb"))
cluster_model = pickle.load(open("model/cluster_model.pkl", "rb"))

def predict_all(data):

    income = data[0]
    loan_amt = data[1]
    cibil = data[2]

    # Loan model (4 features)
    loan_income_ratio = loan_amt / income if income != 0 else 0
    final_data = np.array([[
        income,
        loan_amt,
        cibil,
        loan_income_ratio
    ]])

    loan = loan_model.predict(final_data)[0]
    # Rule-based override
    # if cibil < 550 or loan_amt > 2 * income:
    #     loan = 0   # Reject

    # KMeans (ONLY 3 features)
    cluster_input = np.array([[income, loan_amt, cibil]])
    segment = cluster_model.predict(cluster_input)[0]

    # Fraud
    fraud = fraud_model.predict(np.array([[loan_amt]]))[0]

    # Rule-based fix
    # if fraud == 0:
    #     fraud= "Low"
    # elif fraud == 1 and cibil > 650:
    #     fraud = "Low"
    # else:
    #     fraud= "High"

    return loan, fraud, segment