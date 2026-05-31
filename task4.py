# Import Libraries

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from mpl_toolkits.mplot3d import Axes3D


# Load Dataset

df = pd.read_csv("Titanic-Dataset.csv")


# Dataset Information

print("Dataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nDataset Information:")
print(df.info())


# Check Missing Values

print("\nMissing Values:")
print(df.isnull().sum())


# Handle Missing Values

df["Age"] = df["Age"].fillna(df["Age"].median())

df["Embarked"] = df["Embarked"].fillna(
    df["Embarked"].mode()[0]
)


# Drop Unnecessary Columns

df = df.drop(
    columns=[
        "PassengerId",
        "Name",
        "Ticket",
        "Cabin"
    ]
)


# Encode Categorical Variables

df["Sex"] = df["Sex"].map(
    {
        "male": 0,
        "female": 1
    }
)

df["Embarked"] = df["Embarked"].map(
    {
        "C": 0,
        "Q": 1,
        "S": 2
    }
)


# Target Variable

y = df["Survived"]


# Full Feature Model

X_full = df.drop("Survived", axis=1)

X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_full,
    y,
    test_size=0.2,
    random_state=42
)

full_model = LogisticRegression(max_iter=1000)

full_model.fit(
    X_train_full,
    y_train
)


# Full Model Coefficients

print("\nFULL FEATURE MODEL")

for feature, coefficient in zip(
    X_full.columns,
    full_model.coef_[0]
):
    print(feature, ":", coefficient)

print("Intercept :", full_model.intercept_[0])


# 2 Feature Model

features_2d = [
    "Age",
    "Fare"
]

X_2d = df[features_2d]

scaler_2d = StandardScaler()

X_2d_scaled = scaler_2d.fit_transform(X_2d)

X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
    X_2d_scaled,
    y,
    test_size=0.2,
    random_state=42
)

model_2d = LogisticRegression()

model_2d.fit(
    X_train_2d,
    y_train_2d
)


# 2 Feature Coefficients

print("\n2 FEATURE MODEL")

for feature, coefficient in zip(
    features_2d,
    model_2d.coef_[0]
):
    print(feature, ":", coefficient)

print("Intercept :", model_2d.intercept_[0])


# 3 Feature Model

features_3d = [
    "Age",
    "Fare",
    "Pclass"
]

X_3d = df[features_3d]

scaler_3d = StandardScaler()

X_3d_scaled = scaler_3d.fit_transform(X_3d)

X_train_3d, X_test_3d, y_train_3d, y_test_3d = train_test_split(
    X_3d_scaled,
    y,
    test_size=0.2,
    random_state=42
)

model_3d = LogisticRegression()

model_3d.fit(
    X_train_3d,
    y_train_3d
)


# 3 Feature Coefficients

print("\n3 FEATURE MODEL")

for feature, coefficient in zip(
    features_3d,
    model_3d.coef_[0]
):
    print(feature, ":", coefficient)

print("Intercept :", model_3d.intercept_[0])


# Evaluation Function

def evaluate_model(
    model,
    X_test,
    y_test,
    title
):

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred
    )

    recall = recall_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print("\n" + title)

    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)

    print("\nConfusion Matrix")
    print(cm)

    plt.figure(figsize=(5,4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title(
        title + " Confusion Matrix"
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "True Label"
    )

    plt.show()

    return y_prob


# Evaluate Full Model

full_probabilities = evaluate_model(
    full_model,
    X_test_full,
    y_test,
    "Full Feature Model"
)


# Evaluate 2 Feature Model

probabilities_2d = evaluate_model(
    model_2d,
    X_test_2d,
    y_test_2d,
    "2 Feature Model"
)


# Evaluate 3 Feature Model

probabilities_3d = evaluate_model(
    model_3d,
    X_test_3d,
    y_test_3d,
    "3 Feature Model"
)


# 2D Decision Boundary

x_min = X_2d_scaled[:,0].min() - 1
x_max = X_2d_scaled[:,0].max() + 1

y_min = X_2d_scaled[:,1].min() - 1
y_max = X_2d_scaled[:,1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.02),
    np.arange(y_min, y_max, 0.02)
)

Z = model_2d.predict(
    np.c_[xx.ravel(), yy.ravel()]
)

Z = Z.reshape(xx.shape)

plt.figure(figsize=(8,6))

plt.contourf(
    xx,
    yy,
    Z,
    alpha=0.3
)

plt.scatter(
    X_2d_scaled[:,0],
    X_2d_scaled[:,1],
    c=y,
    edgecolors="black"
)

plt.xlabel("Age (Scaled)")
plt.ylabel("Fare (Scaled)")
plt.title("2D Decision Boundary")

plt.show()


# 3D Decision Plane

fig = plt.figure(figsize=(10,8))

ax = fig.add_subplot(
    111,
    projection="3d"
)

ax.scatter(
    X_3d_scaled[:,0],
    X_3d_scaled[:,1],
    X_3d_scaled[:,2],
    c=y
)

coef = model_3d.coef_[0]

intercept = model_3d.intercept_[0]

x_range = np.linspace(
    X_3d_scaled[:,0].min(),
    X_3d_scaled[:,0].max(),
    10
)

y_range = np.linspace(
    X_3d_scaled[:,1].min(),
    X_3d_scaled[:,1].max(),
    10
)

xx, yy = np.meshgrid(
    x_range,
    y_range
)

zz = -(
    coef[0] * xx +
    coef[1] * yy +
    intercept
) / coef[2]

ax.plot_surface(
    xx,
    yy,
    zz,
    alpha=0.4
)

ax.set_xlabel("Age")
ax.set_ylabel("Fare")
ax.set_zlabel("Pclass")

plt.title("3D Decision Plane")

plt.show()


# New Data Points

data_point_1 = {
    "Pclass": 3,
    "Sex": 0,
    "Age": 22,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked": 2
}

data_point_2 = {
    "Pclass": 1,
    "Sex": 1,
    "Age": 38,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 71.2833,
    "Embarked": 0
}


# Full Model Predictions

new_full = pd.DataFrame(
    [
        data_point_1,
        data_point_2
    ]
)

full_probability = full_model.predict_proba(
    new_full
)

full_prediction = full_model.predict(
    new_full
)

print("\nFULL FEATURE MODEL PREDICTIONS")

for i in range(2):

    print("\nPassenger", i + 1)

    print(
        "Survival Probability:",
        full_probability[i][1]
    )

    print(
        "Predicted Status:",
        full_prediction[i]
    )


# 2 Feature Model Predictions

new_2d = pd.DataFrame(
    [
        [22, 7.25],
        [38, 71.2833]
    ],
    columns=features_2d
)

new_2d = scaler_2d.transform(
    new_2d
)

probability_2d = model_2d.predict_proba(
    new_2d
)

prediction_2d = model_2d.predict(
    new_2d
)

print("\n2 FEATURE MODEL PREDICTIONS")

for i in range(2):

    print("\nPassenger", i + 1)

    print(
        "Survival Probability:",
        probability_2d[i][1]
    )

    print(
        "Predicted Status:",
        prediction_2d[i]
    )


# 3 Feature Model Predictions

new_3d = pd.DataFrame(
    [
        [22, 7.25, 3],
        [38, 71.2833, 1]
    ],
    columns=features_3d
)

new_3d = scaler_3d.transform(
    new_3d
)

probability_3d = model_3d.predict_proba(
    new_3d
)

prediction_3d = model_3d.predict(
    new_3d
)

print("\n3 FEATURE MODEL PREDICTIONS")

for i in range(2):

    print("\nPassenger", i + 1)

    print(
        "Survival Probability:",
        probability_3d[i][1]
    )

    print(
        "Predicted Status:",
        prediction_3d[i]
    )