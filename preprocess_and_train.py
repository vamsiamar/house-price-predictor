import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import joblib

# Load cleaned dataset
df = pd.read_csv("cleaned_makaan_data.csv")

# Drop rows with missing values (if any)
df = df.dropna()

# Filter out 'RK' entries (non-numeric BHK)
df = df[~df['No_of_BHK'].str.contains("RK", na=False)]

# Convert 'No_of_BHK' to numeric
df['No_of_BHK'] = df['No_of_BHK'].str.replace(" BHK", "").astype(int)

# Convert 'Size' to numeric (remove commas and 'sq ft')
df['Size'] = df['Size'].str.replace(" sq ft", "").str.replace(",", "").astype(int)

# Convert 'Price' to numeric (remove commas)
df['Price'] = df['Price'].str.replace(",", "").astype(int)

# Define features and target
X = df.drop("Price", axis=1)
y = df["Price"]

# Feature columns
categorical_cols = ['City_name', 'Locality_Name', 'is_furnished', 'Property_type']
numerical_cols = ['No_of_BHK', 'Size']
binary_cols = ['is_Apartment', 'is_ready_to_move', 'is_RERA_registered']

# ColumnTransformer to encode categorical data
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
], remainder='passthrough')

# Create pipeline
pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('model', LinearRegression())
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
pipeline.fit(X_train, y_train)

# Evaluate
predictions = pipeline.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", round(mse, 2))

# Save model
joblib.dump(pipeline, "model.joblib")
print("Model saved to model.joblib successfully.")
