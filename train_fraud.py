import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data/transactions.csv")

# Use simple feature
X = df[["transaction_amount"]]
y = df["frau.d_flag"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Fraud Model Accuracy:", accuracy)
print("Accuracy:", accuracy * 100, "%")
    
pickle.dump(model, open("model/fraud_model.pkl", "wb"))