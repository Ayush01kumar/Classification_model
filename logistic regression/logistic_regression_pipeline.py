import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

df = pd.read_csv("your_dataset.csv")


# --------------------------------------------------
# Separate features and target
# --------------------------------------------------

X = df.drop(columns=["target"])
y = df["target"]


# --------------------------------------------------
# Identify feature types
# --------------------------------------------------

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns


# --------------------------------------------------
# Numeric preprocessing
# --------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


# --------------------------------------------------
# Categorical preprocessing
# --------------------------------------------------

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


# --------------------------------------------------
# Combine preprocessing
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# --------------------------------------------------
# Complete Logistic Regression pipeline
# --------------------------------------------------

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000
            )
        )
    ]
)


# --------------------------------------------------
# Train
# --------------------------------------------------

model.fit(X, y)


# --------------------------------------------------
# Predictions
# --------------------------------------------------

predictions = model.predict(X)

probabilities = model.predict_proba(X)[:, 1]


print("Predictions:")
print(predictions)

print("\nProbability of class 1:")
print(probabilities)