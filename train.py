import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
file_path = "dataset.csv"
df = pd.read_csv(file_path)

# Drop unnecessary columns
df = df.drop(columns=["Unnamed: 2"], errors='ignore')

# Encode labels (Normal = 0, Abnormal = 1)
df['Labels'] = df['Labels'].map({'Normal': 0, 'Abnormal': 1})

# Split features and labels
X = df[['Values']]
y = df['Labels']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)

# Save the trained model
import joblib
joblib.dump(model, "nerve_signal_model.pkl")
print("Model saved as nerve_signal_model.pkl")
