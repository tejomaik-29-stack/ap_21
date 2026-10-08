import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------
# Create models folder
# ---------------------------------------------------------
os.makedirs("models", exist_ok=True)

# ---------------------------------------------------------
# Generate data
# ---------------------------------------------------------
fahrenheit = np.array([32, 41, 50, 59, 68, 77, 86, 95, 104, 113, 122, 131, 140, 149, 158], dtype=float)

celsius = (5/9) * (fahrenheit - 32)

df = pd.DataFrame({"Fahrenheit": fahrenheit, "Celsius": celsius})

print(df)

# =========================================================
# MACHINE LEARNING MODEL
# =========================================================
f_model = LinearRegression()
f_model.fit(fahrenheit.reshape(-1, 1), celsius)

weight = f_model.coef_[0]
bias = f_model.intercept_

print("\nML Weight:", weight)
print("ML Bias:", bias)
print(f"ML Equation: C = {weight:.4f}F + ({bias:.4f})")

joblib.dump(f_model, "models/f_model.joblib")

# =========================================================
# DEEP LEARNING MODEL
# =========================================================
dl_model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation="relu", input_shape=[1]),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(1)
])

dl_model.compile(optimizer="adam", loss="mean_squared_error")
dl_model.summary()
dl_model.fit(fahrenheit, celsius, epochs=500, verbose=1)

dl_model.save("models/dl_model.keras")

# =========================================================
# TEST PREDICTIONS
# =========================================================
test_values = [32, 50, 68, 86, 104, 122, 212]

print("\nML PREDICTIONS")

for f in test_values:
    actual = (5/9) * (f - 32)
    predicted = f_model.predict(np.array([[f]]))[0]
    print("Data point, Actual Value, Predicted Value")
    print(f"{f}, {actual}, {predicted}")

print("\nDEEP LEARNING PREDICTIONS")

for f in test_values:
    actual = (5/9) * (f - 32)
    predicted = dl_model.predict(np.array([[f]]), verbose=0)[0][0]
    print("Data point, Actual Value, Predicted Value")
    print(f"{f}, {actual}, {predicted}")

print("\nModels saved successfully.")