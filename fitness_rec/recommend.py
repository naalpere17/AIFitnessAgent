import kagglehub
import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

from xgboost import XGBRegressor

#Download the dataset folder
folder_path = kagglehub.dataset_download("nadeemajeedch/fitness-tracker-dataset")
print("Folder path:", folder_path)

#List files to find the CSV name and join path
files = os.listdir(folder_path)
print("Files in folder:", files)
csv_path = os.path.join(folder_path, files[0])

#Load the dataset
df = pd.read_csv(csv_path)

# drop max_bpm since that can cause leakage in the model
df = df.drop(columns=["Max_BPM"])
df = df.dropna(subset=["Avg_BPM", "Calories_Burned"])

print(df.isna().sum().sort_values(ascending=False))

# Split input and expected output columns
target_cols = ["Avg_BPM", "Calories_Burned"]

feature_cols = [
"Age",
"Gender",
"Weight (kg)",
"Height (m)",
"Resting_BPM",
"Session_Duration (hours)",
"Workout_Type",
"Fat_Percentage",
"Water_Intake (liters)",
"Workout_Frequency (days/week)",
"Experience_Level",
"BMI"
]

X = df[feature_cols]
y = df[target_cols]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing Pipeline for numerical and categorical features
# Use OneHotEncoder for categorical features and StandardScaler for numerical features
# so everything is normalized

numeric_features = [
"Age", "Weight (kg)", "Height (m)",
"Resting_BPM", "Session_Duration (hours)",
"Fat_Percentage", "Water_Intake (liters)",
"Workout_Frequency (days/week)", "BMI"
]

categorical_features = [
"Gender", "Workout_Type", "Experience_Level"
]

numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")),("scaler", StandardScaler())])

categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),("encoder", OneHotEncoder(handle_unknown="ignore"))])

preprocessor = ColumnTransformer(transformers=[("num", numeric_transformer, numeric_features),("cat", categorical_transformer, categorical_features)])

# Create the model pipeline: Using XGBRegressor
base_model = XGBRegressor(
n_estimators=400,
max_depth=4,
learning_rate=0.05,
subsample=0.8,
colsample_bytree=0.8,
random_state=42
)

model = MultiOutputRegressor(base_model)

pipeline = Pipeline([
("preprocessing", preprocessor),
("model", model)
])

# Training
pipeline.fit(X_train, y_train)

# Evaluation
preds = pipeline.predict(X_test)

mae_hr = mean_absolute_error(y_test["Avg_BPM"], preds[:, 0])
mae_cal = mean_absolute_error(y_test["Calories_Burned"], preds[:, 1])

r2_hr = r2_score(y_test["Avg_BPM"], preds[:, 0])
r2_cal = r2_score(y_test["Calories_Burned"], preds[:, 1])

print("Heart Rate MAE:", mae_hr)
print("Calories MAE:", mae_cal)
print("Heart Rate R2:", r2_hr)
print("Calories R2:", r2_cal)

# Save the trained model pipeline for later use in the API
joblib.dump(pipeline, "workout_response_model.pkl")


# Check correlation of numeric features with targets (Temporary)

import seaborn as sns
import matplotlib.pyplot as plt


plt.figure(figsize=(10, 8))
sns.heatmap(df[numeric_features + target_cols].corr(), annot=True, cmap='coolwarm')
plt.show()