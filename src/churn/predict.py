import pickle

model_file = "model_C=1.0.bin"


with open(model_file, "rb") as f_in:
    dv, model = pickle.load(f_in)


customer = {
    "customerid": "la-la",
    "gender": "male",
    "seniorcitizen": 0,
    "partner": "no",
    "dependents": "no",
    "tenure": 12,
    "phoneservice": "yes",
    "multiplelines": "yes",
    "internetservice": "fiber_optic",
    "onlinesecurity": "no",
    "onlinebackup": "no",
    "deviceprotection": "no",
    "techsupport": "no",
    "streamingtv": "no",
    "streamingmovies": "yes",
    "contract": "month-to-month",
    "paperlessbilling": "no",
    "paymentmethod": "electronic_check",
    "monthlycharges": 63.05,
    "totalcharges": 1299.3,
}

print(f"Input customer: {customer}\n")

X = dv.transform([customer])
y_pred = model.predict_proba(X)[:, 1]

print(f"Prediction of customer churning: {y_pred}")
