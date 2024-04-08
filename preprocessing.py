import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('train.csv')

print(df.columns.tolist())

# Replace invalid values with 0
invalid = ['#','*','x','A']

columns = [str(i) for i in range(24)]# Column names from "0" to "23"

df[columns] = df[columns].map(lambda x: np.nan if any(i in str(x) for i in invalid) else x)
df[columns] = df[columns].map(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0)
df = df.fillna(0)

# Drop the "Location"
# df = df.drop('Location', axis=1)

# Save the cleaned dataset to a new .csv file
df.to_csv('train_cleaned.csv', index=False)

df = pd.read_csv('test.csv')

# Replace invalid values with 0
df.iloc[:, 2:] = df.iloc[:, 2:].map(lambda x: np.nan if any(i in str(x) for i in invalid) else x)
df.iloc[:, 2:] = df.iloc[:, 2:].map(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0)
df = df.fillna(0)

df.to_csv('test_cleaned.csv', index=False)