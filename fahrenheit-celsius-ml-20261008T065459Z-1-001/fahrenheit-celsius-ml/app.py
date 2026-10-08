from flask import Flask, render_template, request, jsonify

import numpy as np
import joblib
import tensorflow as tf

app = Flask(__name__)

# ---------------------------------------------------------
# Load trained models
# ---------------------------------------------------------
f_model = joblib.load("models/f_model.joblib")

dl_model = tf.keras.models.load_model("models/dl_model.keras")

# ---------------------------------------------------------
# Home page
# ---------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")

# ---------------------------------------------------------
# Prediction API
# ---------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    fahrenheit = float(data["fahrenheit"])
    model_name = data["model"]

    # -----------------------------------------------------
    # ML Prediction
    # -----------------------------------------------------
    if model_name == "ml":
        prediction = f_model.predict(np.array([[fahrenheit]]))[0]
        model_used = "Linear Regression"

    # -----------------------------------------------------
    # DL Prediction
    # -----------------------------------------------------
    elif model_name == "dl":
        prediction = dl_model.predict(np.array([[fahrenheit]]), verbose=0)[0][0]
        model_used = "Deep Learning"

    else:
        return jsonify({"error": "Invalid model"}), 400

    return jsonify({
        "fahrenheit": fahrenheit,
        "celsius": float(prediction),
        "model": model_used
    })

# ---------------------------------------------------------
# Run locally
# ---------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)