import pandas as pd

# Load the dataset with a fallback encoding
df = pd.read_csv("Makaan_Properties_Buy.csv", encoding='ISO-8859-1')

# View shape and basic info
print("Shape:", df.shape)
print("\nColumns:\n", df.columns)
print("\nSample Rows:\n", df.head())
