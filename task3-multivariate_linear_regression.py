import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Load Dataset

df = pd.read_csv("california_housing_train.csv")

print("Dataset Information:\n")
print(df.info())

print("\nColumns in Dataset:")
print(df.columns)

# Feature Selection (Excluding latitude and longitude)

X = df.drop(
    columns=["longitude", "latitude", "median_house_value"]
)

y = df["median_house_value"]

print("\nFeatures Used:")
print(X.columns)

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# Model Training

model = LinearRegression()

model.fit(X_train, y_train)

# Coefficients and Intercept

print("\nLearned Coefficients:")

for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature:20s} : {coef:,.2f}")

print("\nIntercept :", model.intercept_)

# Prediction on Test Set

y_pred = model.predict(X_test)

# Regression Metrics

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\nRegression Metrics")
print("------------------")
print(f"MAE      : ${mae:,.2f}")
print(f"MSE      : {mse:,.2f}")
print(f"RMSE     : ${rmse:,.2f}")
print(f"R² Score : {r2:.4f}")

# Classification Metrics
# Converted prices into two classes:
# Above median -> 1
# Below median -> 0

threshold = y_test.median()

y_test_class = (y_test > threshold).astype(int)

y_pred_class = (y_pred > threshold).astype(int)

accuracy = accuracy_score(
    y_test_class,
    y_pred_class
)

precision = precision_score(
    y_test_class,
    y_pred_class
)

recall = recall_score(
    y_test_class,
    y_pred_class
)

f1 = f1_score(
    y_test_class,
    y_pred_class
)

cm = confusion_matrix(
    y_test_class,
    y_pred_class
)

print("\nClassification Metrics")
print("----------------------")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nConfusion Matrix")
print(cm)

# Actual vs Predicted Plot

plt.figure(figsize=(8,6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.6
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--',
    label="y = x"
)

plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices")
plt.legend()

# Prediction for New Data Points

new_data = pd.DataFrame(
    [
        [41, 880, 129, 322, 126, 8.3252],
        [28, 960, 160, 310, 150, 4.5001]
    ],
    columns=[
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income"
    ]
)

predictions = model.predict(new_data)

print("\nPredicted House Prices")

for i, price in enumerate(predictions, start=1):
    print(f"Data Point {i}: {price}")

plt.show()
