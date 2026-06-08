import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text

# =====================================================
# Part 1: Dataset Loading and Inspection
# =====================================================

df = pd.read_csv("PlayTennis.csv")

print("Dataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nClass Distribution:")
print(df["play"].value_counts())

target_col = "play"
features = ["outlook", "temp", "humidity", "windy"]

print("\nTarget Column:")
print(target_col)

print("\nNumber of Predictor Features:")
print(len(features))

for feature in features:
    print(f"\nUnique values in {feature}:")
    print(df[feature].unique())

# =====================================================
# Part 2: Entropy of Dataset
# =====================================================

def entropy(labels):

    probs = labels.value_counts(normalize=True)

    return -sum(
        p * np.log2(p)
        for p in probs
    )


dataset_entropy = entropy(df[target_col])

print("\nDataset Entropy:")
print(dataset_entropy)

# =====================================================
# Part 3: Information Gain
# =====================================================

def information_gain(df, feature, target):

    total_entropy = entropy(df[target])

    weighted_entropy = 0

    for value in df[feature].unique():

        subset = df[df[feature] == value]

        weight = len(subset) / len(df)

        weighted_entropy += (
            weight * entropy(subset[target])
        )

    return total_entropy - weighted_entropy


print("\nInformation Gain Table")

ig_scores = {}

for feature in features:

    ig = information_gain(
        df,
        feature,
        target_col
    )

    ig_scores[feature] = ig

    print(f"{feature}: {ig:.4f}")


best_root = max(
    ig_scores,
    key=ig_scores.get
)

print("\nBest Root Attribute:")
print(best_root)

# =====================================================
# Part 4: One Recursive Split
# =====================================================

sunny_subset = df[
    df["outlook"] == "sunny"
]

print("\nSunny Subset:")
print(sunny_subset)

print("\nSunny Subset Entropy:")
print(
    entropy(
        sunny_subset[target_col]
    )
)

remaining_features = [
    f
    for f in features
    if f != "outlook"
]

print("\nInformation Gain Inside Sunny Subset")

for feature in remaining_features:

    ig = information_gain(
        sunny_subset,
        feature,
        target_col
    )

    print(f"{feature}: {ig:.4f}")

# =====================================================
# Part 5: Gini Impurity
# =====================================================

def gini(labels):

    probs = labels.value_counts(
        normalize=True
    )

    return 1 - sum(
        p ** 2
        for p in probs
    )


dataset_gini = gini(
    df[target_col]
)

print("\nDataset Gini:")
print(dataset_gini)

print("\nWeighted Gini Table")

gini_scores = {}

for feature in features:

    weighted_gini = 0

    for value in df[feature].unique():

        subset = df[
            df[feature] == value
        ]

        weight = len(subset) / len(df)

        weighted_gini += (
            weight *
            gini(subset[target_col])
        )

    gini_scores[feature] = weighted_gini

    print(
        f"{feature}: {weighted_gini:.4f}"
    )


best_gini_feature = min(
    gini_scores,
    key=gini_scores.get
)

print("\nBest Root Attribute by Gini:")
print(best_gini_feature)

# =====================================================
# Part 6: Verification Using Scikit-Learn
# =====================================================

X = df[features].copy()

for col in X.columns:

    le = LabelEncoder()

    X[col] = le.fit_transform(
        X[col]
    )

y_encoder = LabelEncoder()

y = y_encoder.fit_transform(
    df[target_col]
)

# Entropy Tree

tree_entropy = DecisionTreeClassifier(
    criterion="entropy",
    random_state=42
)

tree_entropy.fit(X, y)

# Gini Tree

tree_gini = DecisionTreeClassifier(
    criterion="gini",
    random_state=42
)

tree_gini.fit(X, y)

print("\nDecision Tree Rules (Entropy):")

print(
    export_text(
        tree_entropy,
        feature_names=list(X.columns)
    )
)

print("\nDecision Tree Rules (Gini):")

print(
    export_text(
        tree_gini,
        feature_names=list(X.columns)
    )
)

# =====================================================
# Part 7: Final Reflection
# =====================================================

print("\nReflection")

print(
    "\nEntropy measures uncertainty using logarithms."
)

print(
    "Gini measures impurity using squared probabilities."
)

print(
    "Decision Trees split recursively because one split is usually insufficient to classify all samples."
)

print(
    "The best root attribute found manually should match the Scikit-Learn result."
)
