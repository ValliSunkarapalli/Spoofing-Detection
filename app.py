import matplotlib
matplotlib.use('Agg')  # Fix for Matplotlib GUI warning
import matplotlib.pyplot as plt

from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import os
import time
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load trained model and scaler
model = tf.keras.models.load_model("models/spoofing_detector.h5")
scaler = joblib.load("models/scaler.pkl")
expected_features = joblib.load("models/feature_names.pkl")  # Load feature names from training

# Store history
prediction_history = []
recovery_log = []
label_encoders = {}

# 🏠 Home Page
@app.route('/')
def home():
    return render_template('index.html')

# 📂 Upload & Process CSV
@app.route('/upload', methods=['POST'])
def upload():
    try:
        file = request.files['file']
        if not file:
            return render_template('index.html', prediction_text="No file uploaded.", images=[])

        # Load dataset
        df = pd.read_csv(file)

        # 🔹 Ensure dataset matches expected features
        df = adjust_features(df, expected_features)

        # Preprocess data
        features_scaled = scaler.transform(df)

        # Make predictions
        predictions = model.predict(features_scaled)
        results = []
        
        for i, pred in enumerate(predictions.flatten()):
            if pred > 0.5:
                action = initiate_recovery(i)
                results.append(f"Spoofing Attack Detected - {action}")
            else:
                results.append("Normal Activity")

        prediction_history.extend(predictions.flatten())

        # 📊 Generate graphs (fixed the missing function)
        generate_graphs()

        # Save results
        df['Prediction'] = results
        output_csv = "static/predictions.csv"
        df.to_csv(output_csv, index=False)

        return render_template('index.html', prediction_text="Predictions completed! Download results below.", 
                               images=["static/spoofing_trend.png", "static/attack_distribution.png"],
                               download_link=output_csv, recovery_log=recovery_log)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}", images=[])

# 📌 Function to Adjust Features
def adjust_features(df, expected_features):
    """
    Ensures dataset has the same features as the model expects.
    - Removes extra columns.
    - Adds missing columns with default values (0).
    - Converts categorical features using Label Encoding.
    """
    categorical_cols = df.select_dtypes(include=['object']).columns

    # Encode categorical features
    for col in categorical_cols:
        if col not in label_encoders:
            label_encoders[col] = LabelEncoder()
            df[col] = label_encoders[col].fit_transform(df[col])
        else:
            df[col] = df[col].map(lambda s: label_encoders[col].transform([s])[0] if s in label_encoders[col].classes_ else -1)

    # Keep only expected features
    df = df.reindex(columns=expected_features, fill_value=0)  # Add missing columns with default 0

    return df

# 📊 Generate Graphs (Fix for missing function)
def generate_graphs():
    os.makedirs("static", exist_ok=True)

    # 📈 1. Spoofing Detection Trend (Line Graph)
    plt.figure(figsize=(8, 4))
    if prediction_history:
        x = range(1, len(prediction_history) + 1)
        plt.plot(x, prediction_history, marker='o', linestyle='-', color='b', label="Prediction Score")
        plt.axhline(y=0.5, color='r', linestyle='--', label="Threshold (0.5)")
        plt.xlabel("Sample Number")
        plt.ylabel("Spoofing Probability")
        plt.title("Spoofing Detection Trend")
        plt.legend()
        plt.grid(True)
        plt.savefig("static/spoofing_trend.png")
    plt.close()

    # 📊 2. Attack Probability Distribution (Histogram)
    plt.figure(figsize=(8, 4))
    if prediction_history:
        plt.hist(prediction_history, bins=10, color='purple', alpha=0.7, edgecolor='black')
        plt.xlabel("Spoofing Probability")
        plt.ylabel("Frequency")
        plt.title("Attack Probability Distribution")
        plt.grid(True)
        plt.savefig("static/attack_distribution.png")
    plt.close()

# 🚨 Recovery Function
def initiate_recovery(sample_id):
    action = f"Account {sample_id} frozen & user alerted."
    recovery_log.append(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {action}")
    return action

if __name__ == "__main__":
    app.run(debug=True)
