import joblib
from sklearn.metrics import accuracy_score

model = joblib.load("savedmodel.pth")
X_test, y_test = joblib.load("test_data.pkl")

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"Test Accuracy: {accuracy:.4f}")
