import pandas as pd

# Load the collected gesture data
df = pd.read_csv("hand_signs.csv")

# Check if the 'label' column exists
if "label" not in df.columns:
    raise ValueError("❌ 'label' column not found in CSV. Check your file.")

# Count samples per gesture label
label_counts = df["label"].value_counts()

# Display results
print("✅ Samples per gesture label:\n")
print(label_counts)

# Total number of collected samples
total_samples = len(df)
print(f"\n📦 Total samples collected: {total_samples}")