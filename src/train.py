import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Load dataset
df = pd.read_csv('iris.csv')
X = df.drop(columns='species')
y = df['species']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train KNN
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'knn_model.pkl')
print("Model trained and saved.")
