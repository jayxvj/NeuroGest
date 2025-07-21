import pandas as pd

# Load CSV
df = pd.read_csv("pose_data.csv")

# Show all unique labels
print("✅ Available signs:", df['label'].unique())

# Sign you want to delete
sign_to_remove = input("Enter the label you want to delete: ")

# Filter out the sign
df_filtered = df[df['label'] != sign_to_remove]

# Save back
df_filtered.to_csv("pose_data.csv", index=False)

print(f"✅ All samples for '{sign_to_remove}' have been removed.")
print("Remaining sign counts:\n", df_filtered['label'].value_counts())