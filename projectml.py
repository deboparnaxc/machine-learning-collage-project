
# Parkinson's Disease Prediction Using ANN

# 1. Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# # 2. Load the dataset
# print("Loading Parkinson's Disease Dataset...")

# parkinsons = fetch_ucirepo(id=174)

# X = parkinsons.data.features.copy()
# y = parkinsons.data.targets["status"].astype(int)

# # 3. Store patient IDs for group-based splitting
# patient_groups = X["name"].copy()

# # Remove unnecessary ID column
# X = X.drop(columns=["name"])
# Load original UCI dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"

df = pd.read_csv(url)

# Separate features and target
patient_groups = df["name"].copy()

X = df.drop(columns=["name", "status"])
y = df["status"].astype(int)

# 4. Check dataset
print("\nDataset Information:")
print("Total Records:", len(X))
print("Total Features:", X.shape[1])
print("\nMissing Values:", X.isnull().sum().sum())
print("\nTarget Distribution:")
print(y.value_counts())

# 5. Split data by patient
# No patient's recordings will appear in both sets

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

train_index, test_index = next(
    cv.split(X, y, groups=patient_groups)
)

X_train = X.iloc[train_index]
X_test = X.iloc[test_index]

y_train = y.iloc[train_index]
y_test = y.iloc[test_index]

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))

# 6. Build ANN model
model = Pipeline([
    ("scaler", StandardScaler()),

    ("ann", MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=1500,
        random_state=42,
        early_stopping=False
    ))
])

# 7. Train the ANN model
print("\nTraining ANN Model...")

model.fit(X_train, y_train)

print("Model Training Completed!")

# 8. Predict test data
y_pred = model.predict(X_test)

# 9. Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========")

print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Healthy", "Parkinson's"],
    zero_division=0
))

# 10. Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Healthy", "Parkinson's"]
)

display.plot(cmap="Blues")
plt.title("Parkinson's Disease - ANN Confusion Matrix")
plt.show()

# 11. Plot actual vs predicted values
plt.figure(figsize=(8, 4))

plt.plot(
    range(len(y_test)),
    y_test.to_numpy(),
    marker="o",
    label="Actual"
)

plt.plot(
    range(len(y_pred)),
    y_pred,
    marker="x",
    label="Predicted"
)

plt.xlabel("Test Record")
plt.ylabel("Class")
plt.yticks([0, 1], ["Healthy", "Parkinson's"])
plt.title("Actual vs Predicted Results")
plt.legend()
plt.grid(True)
plt.show()

# 12. Save the trained model
joblib.dump(model, "parkinsons_ann_model.pkl")

print("\nTrained model saved successfully!")

# 13. Demonstrate prediction on a held-out record
sample = X_test.iloc[[0]]

prediction = model.predict(sample)[0]
probability = model.predict_proba(sample)[0]

print("\n========== SAMPLE PREDICTION ==========")

if prediction == 1:
    print("Predicted Class: Parkinson's")
else:
    print("Predicted Class: Healthy")

print("Healthy class score:", round(probability[0] * 100, 2), "%")
print("Parkinson's class score:", round(probability[1] * 100, 2), "%")

print("\nProject Execution Completed!")
