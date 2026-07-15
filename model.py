import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

TARGET_COLUMN = "Approved"
CATEGORICAL_COLUMNS = [
    "Gender",
    "Education",
    "Marital Status",
    "Home Status",
    "Employment Type",
    "Region",
    "Industry",
    "Has Mortgage",
    "Prior Default",
]
NUMERICAL_COLUMNS = [
    "Age",
    "Annual Income($)",
    "Employment Years",
    "Months At Address",
    "Credit Score",
    "Num Credit Cards",
    "Credit Utilization(%)",
    "Num Late Payments",
    "Num Inquiries 6M",
    "Monthly Debt($)",
    "Savings Balance($)",
]


def load_data(path: str = "credit.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def build_pipeline() -> Pipeline:
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent", fill_value="Missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )
    numerical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )
    transformer = ColumnTransformer(
        [
            ("num", numerical_pipeline, NUMERICAL_COLUMNS),
            ("cat", categorical_pipeline, CATEGORICAL_COLUMNS),
        ]
    )
    classifier = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    pipeline = Pipeline(
        [
            ("transformer", transformer),
            ("classifier", classifier),
        ]
    )
    return pipeline


def train_model(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    x = df[CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS]
    y = df[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, stratify=y, random_state=random_state
    )
    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, pipeline.predict_proba(x_test)[:, 1]),
        "report": classification_report(y_test, y_pred, digits=4),
    }
    return pipeline, results


def predict(pipeline: Pipeline, feature_dict: dict) -> dict:
    df = pd.DataFrame([feature_dict])
    proba = pipeline.predict_proba(df)[0, 1]
    label = int(pipeline.predict(df)[0])
    return {"label": label, "probability": float(proba), "input": df}


def build_default_input(df: pd.DataFrame) -> dict:
    defaults = {}
    for feature in CATEGORICAL_COLUMNS:
        defaults[feature] = df[feature].mode(dropna=True)[0]
    for feature in NUMERICAL_COLUMNS:
        defaults[feature] = float(df[feature].median(skipna=True))
    return defaults
