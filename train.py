import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from detection import build_model

# 🔹 Function to load and preprocess dataset
def load_and_preprocess(dataset_paths):
    """
    Loads datasets, preprocesses them, and returns training & testing data.
    Automatically detects categorical and numerical features.
    """
    all_data = []

    for path in dataset_paths:
        df = pd.read_csv(path)
        all_data.append(df)

    # Merge datasets
    full_df = pd.concat(all_data, axis=0, ignore_index=True)

    # 🔹 Identify categorical columns and encode them
    categorical_cols = full_df.select_dtypes(include=['object']).columns
    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        full_df[col] = le.fit_transform(full_df[col])
        label_encoders[col] = le  # Store encoders for future use

    # 🔹 Select features (all columns except the target column)
    X = full_df.iloc[:, :-1]
    y = full_df.iloc[:, -1]

    # Standardize numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split dataset into training & testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 🔹 Save feature names for use during prediction
    feature_names = X.columns.tolist()
    joblib.dump(feature_names, "models/feature_names.pkl")

    print(f"✅ Data Preprocessing Complete: {X_train.shape[1]} features selected.")

    return X_train, X_test, y_train, y_test, scaler

# 🔹 Define dataset paths
dataset_paths = ["data/UNSW_NB15_testing-set.csv", "data/UNSW_NB15_training-set.csv"]

# Load and preprocess dataset
X_train, X_test, y_train, y_test, scaler = load_and_preprocess(dataset_paths)

# Build and train model
model = build_model(X_train.shape[1])
model.fit(X_train, y_train, epochs=25, batch_size=32, validation_data=(X_test, y_test))

# Save trained model and scaler
os.makedirs("models", exist_ok=True)
model.save("models/spoofing_detector.h5")
joblib.dump(scaler, "models/scaler.pkl")

print("✅ Model training complete. Saved as spoofing_detector.h5")
