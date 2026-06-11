import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data/loan_applications.csv")
df["loan_income_ratio"] = df["loan_amount_requested"] / df["monthly_income"]

#  Select correct columns
X = df[["monthly_income", "loan_amount_requested", "cibil_score","loan_income_ratio"]]
y = df["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Loan Model Accuracy:", accuracy)
print("Accuracy:", accuracy * 100, "%")


pickle.dump(model, open("model/loan_model.pkl", "wb"))