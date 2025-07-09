import joblib
import numpy as np

df = pd.read_csv('iris.data.csv', header=None)
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']


# Load model and label encoder
model = joblib.load('knn_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Example input: sepal_length, sepal_width, petal_length, petal_width
sample_input = np.array([[5.1, 3.5, 1.4, 0.2]])

# Predict
predicted_class = model.predict(sample_input)[0]
predicted_label = label_encoder.inverse_transform([predicted_class])[0]

print(f"Predicted Iris class: {predicted_label}")
