import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Read the CSV file
df = pd.read_csv('model/keypoint_classifier/keypoint2.csv', header=None)

# Initialize the MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))

# Normalize the data (excluding the first column if it's a label)
df.iloc[:, 1:] = scaler.fit_transform(df.iloc[:, 1:])

# Save the normalized data back to a CSV file
df.to_csv('model/keypoint_classifier/normalized_keypoint2.csv', header=False, index=False)

print("Normalization complete. Saved to 'normalized_keypoint2.csv'.")