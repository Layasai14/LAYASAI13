import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv('iris.data.csv', header=None)
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

# Encode target labels
label_encoder = LabelEncoder()
df['class'] = label_encoder.fit_transform(df['class'])

# Split data
X = df.drop('class', axis=1)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Save model and label encoder
joblib.dump(model, 'knn_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

# Optional: evaluate model
accuracy = model.score(X_test, y_test)
print(f"Model trained with accuracy: {accuracy:.2f}")
