import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

# Load synthetic dataset
df = pd.read_csv("synthetic_fitness_dataset.csv")

# Constants for training
# These match the headers provided in your dataset
target_col = "Avg_BPM" 
feature_cols = ["Age", "BMI", "Resting_BPM", "Weight (kg)", "Session_Duration (hours)", "Calories_Burned"]

# Adding placeholders for personal features to maintain pipeline shape
df["acwr"] = 1.0
df["readiness"] = 0.5
df["acute_load"] = df["Calories_Burned"]
df["chronic_load"] = df["Calories_Burned"]
df["est_max_bpm"] = 220 - df["Age"]

full_features = feature_cols + ["acwr", "readiness", "acute_load", "chronic_load", "est_max_bpm"]

X = df[full_features]
y = df[target_col]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), full_features)]
)

global_model = Pipeline([
    ("preprocessing", preprocessor),
    ("model", XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=42))
])

global_model.fit(X_train, y_train)
joblib.dump(global_model, "global_intensity_model.pkl")
print("Global model saved as 'global_intensity_model.pkl'")