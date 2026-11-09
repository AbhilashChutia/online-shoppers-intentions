import pickle
import xgboost as xgb
from flask import Flask
from flask import request
from flask import jsonify

model_file = "model.bin"

with open(model_file, "rb") as f_in:
    dv, model = pickle.load(f_in)

app = Flask("shopper")


@app.route("/predict", methods=["POST"])
def predict():
    shopper = request.get_json()

    X = dv.transform([shopper])
    features = list(dv.get_feature_names_out())
    dX = xgb.DMatrix(X, feature_names=features)

    y_pred = model.predict(dX)[0]
    purchase = y_pred >= 0.5

    result = {"purchase_probability": float(y_pred), "purchase": bool(purchase)}

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
