import pandas as pd

# Load the raw dataset with correct encoding
df = pd.read_csv("Makaan_Properties_Buy.csv", encoding='ISO-8859-1')

# Select useful columns
selected_cols = [
    'City_name', 'Locality_Name', 'No_of_BHK', 'Size', 'is_furnished',
    'Property_type', 'is_Apartment', 'is_ready_to_move',
    'is_RERA_registered', 'Price'
]
df = df[selected_cols]

# Drop missing values
df.dropna(inplace=True)

# Save cleaned data
df.to_csv("cleaned_makaan_data.csv", index=False)

print("Cleaned data saved as 'cleaned_makaan_data.csv'.")
print("New shape:", df.shape)
print(df.head())
