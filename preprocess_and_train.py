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

# Drop 'RK' rows and clean numeric fields
df = df[~df['No_of_BHK'].str.contains("RK", na=False)]
df['No_of_BHK'] = df['No_of_BHK'].str.replace(" BHK", "").astype(int)
df['Size'] = df['Size'].str.replace(" sq ft", "").str.replace(",", "").astype(int)
df['Price'] = df['Price'].str.replace(",", "").astype(int)

# 🧹 Drop rare Localities (less than 30 listings)
top_localities = df['Locality_Name'].value_counts()[df['Locality_Name'].value_counts() > 30].index
df = df[df['Locality_Name'].isin(top_localities)]

# Prepare data
X = df.drop("Price", axis=1)
y = df["Price"]

categorical_cols = ['City_name', 'Locality_Name', 'is_furnished', 'Property_type']
binary_cols = ['is_Apartment', 'is_ready_to_move', 'is_RERA_registered']
numerical_cols = ['No_of_BHK', 'Size']

# 🚀 Encode only top 50 most frequent categories
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown='ignore', max_categories=50, sparse_output=False), categorical_cols),
], remainder='passthrough')

pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", LinearRegression())
])

print("Splitting and training...")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_test)
mse = mean_squared_error(y_test, predictions)

print("✅ Mean Squared Error:", round(mse, 2))
joblib.dump(pipeline, "model.joblib")
print("✅ Model saved to model.joblib successfully.")
