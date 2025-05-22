import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_preprocess(dataset_paths):
    """
    Loads multiple datasets, preprocesses them, and returns training & testing data.
    Args:
        dataset_paths (list): List of dataset file paths.
    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    all_data = []

    # Load each dataset
    for path in dataset_paths:
        df = pd.read_csv(path)
        all_data.append(df)

    # Merge datasets
    full_df = pd.concat(all_data, axis=0, ignore_index=True)

    # Identify categorical columns
    categorical_cols = full_df.select_dtypes(include=['object']).columns

    # Encode categorical features using Label Encoding
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        full_df[col] = le.fit_transform(full_df[col])
        label_encoders[col] = le  # Store encoders for later use

    # Assume the last column is the target variable (0 = Normal, 1 = Spoofing)
    X = full_df.iloc[:, :-1]  # Features
    y = full_df.iloc[:, -1]   # Target label

    # Standardize numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split dataset into training & testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Print feature count for debugging
    print(f"✅ Data Loaded & Preprocessed: {X_train.shape[1]} features")

    return X_train, X_test, y_train, y_test, scaler
