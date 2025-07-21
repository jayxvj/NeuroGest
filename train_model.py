# --- train_model.py ---
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Step 1: Fix header if missing
csv_path = "hand_signs.csv"
if not os.path.isfile(csv_path):
    raise FileNotFoundError("❌ CSV file not found. Make sure 'hand_signs.csv' exists.")

# Load raw CSV and fix header if needed
df_raw = pd.read_csv(csv_path, header=None)
if 'label' not in df_raw.iloc[0].values:
    columns = [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)] + ["label"]
    df_raw.columns = columns
    df_raw = df_raw[1:]  # remove previous first row
    df_raw.to_csv(csv_path, index=False)
    print("✅ Fixed CSV header.")

# Step 2: Load cleaned data
df = pd.read_csv(csv_path)
if 'label' not in df.columns:
    raise ValueError("❌ 'label' column still not found. Check your CSV file manually.")

print("[INFO] Label counts:\n", df['label'].value_counts())

X = df.drop("label", axis=1)
y = df["label"]

# Step 3: Train-test split
if len(df) < 2:
    raise ValueError("❌ Not enough data to train. Collect more samples.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train model
model = KNeighborsClassifier(n_neighbors=min(3, len(X_train)))
model.fit(X_train, y_train)

# Step 5: Evaluate
y_pred = model.predict(X_test)
print("[INFO] Accuracy:", accuracy_score(y_test, y_pred))
print("[INFO] Report:\n", classification_report(y_test, y_pred))

# Step 6: Save model
joblib.dump(model, "hand_sign_model.pkl")
print("✅ Model saved as 'hand_sign_model.pkl'")