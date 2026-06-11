import pickle
import pandas as pd
from sklearn.cluster import KMeans

df = pd.read_csv("data/loan_applications.csv")

# SAME FEATURES
X = df[["monthly_income", "loan_amount_requested", "cibil_score"]]

model = KMeans(n_clusters=3)
model.fit(X)

pickle.dump(model, open("model/cluster_model.pkl", "wb"))