import joblib
import pandas as pd

# Load model
model = joblib.load('knn_model.pkl')

# Example input data
sample = pd.DataFrame({
    'sepal_length': [5.1],
    'sepal_width': [3.5],
    'petal_length': [1.4],
    'petal_width': [0.2]
})

# Predict
prediction = model.predict(sample)
print("Predicted species:", prediction[0])
