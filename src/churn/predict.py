import pickle

from flask import Flask, request, jsonify

app = Flask('ChurnService')

model_file = "model_C=1.0.bin"

@app.route('/churn', methods=["POST"])
def exec():
    with open(model_file, "rb") as f_in:
        dv, model = pickle.load(f_in)

    customer = request.get_json()
    print(f"Input customer: {customer}\n")

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[:, 1]

    churn = (y_pred >= 0.5)
    print(f"Prediction of customer churning: {y_pred}")

    response = {
        "churn_prediction_val": float(y_pred),
        "is_churning": bool(churn)
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=9696)
