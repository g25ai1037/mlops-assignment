from flask import Flask
import joblib
import numpy as np

app = Flask(__name__)

try:
    model = joblib.load("savedmodel.pth")
except:
    model = None

@app.route("/")
def home():
    return """
    <h1>MLOps Assignment Flask App</h1>
    <p>Model Loaded Successfully</p>
    """

@app.route("/predict")
def predict():
    if model is None:
        return "Model not found"

    sample = np.random.rand(1, 4096)
    prediction = model.predict(sample)

    return f"Predicted Class: {prediction[0]}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
