from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from preprocess import load_and_preprocess
import tensorflow as tf
import numpy as np

# Load preprocessed data
dataset_paths = ["data/UNSW-NB15.csv", "data/IoT-Spoofing.csv"]
X_train, X_test, y_train, y_test, _ = load_and_preprocess(dataset_paths)

# Load trained model
model = tf.keras.models.load_model("models/spoofing_detector.h5")

# Make predictions
y_pred = (model.predict(X_test) > 0.5).astype(int)

# Print evaluation metrics
print("✅ Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
