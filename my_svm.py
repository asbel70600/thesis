# from inspect import ClassFoundException
# from math import e, nan
from sklearn.svm import SVC

# from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier

# import numpy as np
from sklearn.impute import SimpleImputer
from time import time

# from sklearn.model_selection import GridSearchCV

prelude = "."
input_path = f"{prelude}/data/train/data.csv"
eso = ["anita", "atlas", "metro", "alouette", "manitov"]
a = []
for i in eso:
    df = pd.read_csv(f"./data/data_{i}.csv", index_col=False).dropna()
    a.append(df)


full_ds = pd.read_csv(input_path, index_col=False)
merged_ds = pd.concat(a, ignore_index=True)
merged_ds = merged_ds.dropna()
full_ds = merged_ds.sample(frac=1, random_state=42).reset_index(drop=True)
full_ds = full_ds.dropna()
features = full_ds.drop(columns=["class"]).astype(float).to_numpy()
classes = full_ds["class"].to_numpy()
len = int(0.25 * len(full_ds["class"]))


# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(features, classes)
#
# # Step 3: Get feature importances
# importances = model.feature_importances_
# indexes = model.feature_names_in_
#
# # Step 4: Create a DataFrame to hold feature importances
# feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
#
# # Step 5: Filter features based on importance (e.g., keep features with importance greater than a threshold)
# threshold = 0.1  # Set your threshold for feature importance
# important_features = feature_importances[feature_importances['Importance'] > threshold]
#
# print("Important Features:")
# print(important_features)
#
# # Optional: Selecting only important features for further analysis
# selected_features = important_features['Feature'].tolist()
# X_selected = df[selected_features]


# features, testing_features, classes, testing_classes = train_test_split(
#     features, classes, test_size=0.2, random_state=round(time())
# )
#
# # Create an imputer object with a chosen strategy
# imputer = SimpleImputer(strategy="mean")  # or 'median', 'most_frequent', etc.
#
# # Fit the imputer on the features and transform them
# features = imputer.fit_transform(features)
# testing_features = imputer.transform(testing_features)
#
# # Now scale the features
# scaler = StandardScaler()
# features = scaler.fit_transform(features)
# testing_features = scaler.transform(testing_features)
#
# scaler = StandardScaler()
# features = scaler.fit_transform(features)
# testing_features = scaler.transform(testing_features)
#
# for i, arr in enumerate(features):
#     if any(pd.isna(x) for x in arr):
#         features = np.delete(features, i)
#         classes = np.delete(classes, i)
#
# for i, arr in enumerate(testing_features):
#     if any(pd.isna(x) for x in arr):
#         testing_features = np.delete(testing_features, i)
#         testing_classes = np.delete(testing_classes, i)
#
# model = SVC(
#     C=10,
#     kernel="rbf",
#     gamma=0.00007,
#     coef0=0.0,
#     degree=2,
#     cache_size=1000,
# )
# # model.gamma = 0.0095
# # model = SVC()
# model.fit(features, classes)
#
# well_class = 0
# bad_class = 0
#
# for i, j in enumerate(features):
#     response = model.predict([j])
#     if response[0] == classes[i]:
#         well_class += 1
#     else:
#         bad_class += 1
#
# bias = (bad_class * 100) / (bad_class + well_class)
# well_class = 0
# bad_class = 0
#
# for i, j in enumerate(testing_features):
#     response = model.predict([j])
#     if response[0] == testing_classes[i]:
#         well_class += 1
#     else:
#         bad_class += 1
#
# fit = (bad_class * 100) / (bad_class + well_class)
# x = abs(bias - fit)
# weight = 1 / (1 + (e ** (2 * x - 4)))
# new_score = weight * ((bias + fit) / 2)
#
# print(f"bias: {bias} fit: {fit}")
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, accuracy_score

# Split data
features, testing_features, classes, testing_classes = train_test_split(
    features, classes, test_size=0.3, random_state=round(time())
)

# Create a pipeline for preprocessing and model training
pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="mean")),  # Handle missing values
        ("scaler", StandardScaler()),  # Scale features
        (
            "svm",
            BaggingClassifier(
                estimator=SVC(
                    C=10,
                    kernel="rbf",
                    gamma='',
                    coef0=0.0,
                    degree=2,
                    cache_size=1000,
                ), n_estimators=10
            ),
        ),  # Model
    ]
)

# Perform 5-fold cross-validation
cv_scores = cross_val_score(pipeline, features, classes, cv=5, scoring="accuracy")



# Print CV results
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean()}")

# Train on the full training set
pipeline.fit(features, classes)

# Evaluate on testing set
predictions = pipeline.predict(testing_features)
test_accuracy = accuracy_score(testing_classes, predictions)

print(f"Test Accuracy: {test_accuracy}")
