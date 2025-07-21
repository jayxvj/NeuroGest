import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from joblib import dump

# Load the data
df = pd.read_csv("pose_data.csv")

# Separate features and label
X = df.drop("label", axis=1)
y = df["label"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a KNN model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("📊 Model Evaluation:\n")
print(classification_report(y_test, y_pred))

# Save the trained model
dump(model, "activity_model.pkl")
print("\n✅ Trained model saved as 'activity_model.pkl'")