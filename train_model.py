import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import joblib

# Load cleaned dataset
df = pd.read_excel("cleaned_premiums.xlsx")

# Define features and target
X = df.drop("Annual_Premium_Amount", axis=1)
y = df["Annual_Premium_Amount"]

# Identify categorical columns
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)],
    remainder="passthrough"
)

# Pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", XGBRegressor(random_state=42))
])

# Train and save
pipeline.fit(X, y)
joblib.dump(pipeline, "xgb_model.pkl")
joblib.dump(X.columns.tolist(), "input_columns.pkl")
print("✅ Model and columns saved.")
