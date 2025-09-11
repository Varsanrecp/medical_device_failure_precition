# train_model.py
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report

def train_and_save(excel_path=None, models_dir=None):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if models_dir is None:
        models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    if excel_path is None:
        excel_path = os.environ.get("EXCEL_PATH") or os.path.join(base_dir, "final_cts.xlsx")

    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel not found at {excel_path}. Put final_cts.xlsx in repo or set EXCEL_PATH.")

    df = pd.read_excel(excel_path)

    # Choose the features the web form is supplying (keep consistent with app)
    FEATURES = ["classification", "code", "implanted", "name_device", "name_manufacturer"]

    # Ensure features exist in df. If missing, create column filled with 'Unknown'
    for c in FEATURES:
        if c not in df.columns:
            df[c] = "Unknown"

    # Fill missing risk_class and features
    if "risk_class" not in df.columns:
        raise ValueError("Column 'risk_class' not found in dataset.")
    df["risk_class"] = df["risk_class"].fillna(df["risk_class"].mode()[0]).astype(int)
    df[FEATURES] = df[FEATURES].fillna("Unknown").astype(str)

    # Build label-to-int mappings per categorical column (ensure 'Unknown' present)
    mappings = {}
    X = pd.DataFrame()
    for col in FEATURES:
        unique_vals = list(df[col].unique())
        if "Unknown" not in unique_vals:
            unique_vals = ["Unknown"] + [v for v in unique_vals if v != "Unknown"]
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        mappings[col] = mapping
        X[col] = df[col].map(mapping)

    y = df["risk_class"].astype(int)

    # Imputer to be safe (most_frequent)
    imputer = SimpleImputer(strategy="most_frequent")
    imputer.fit(X)
    X_imputed = pd.DataFrame(imputer.transform(X), columns=X.columns)

    # Train-test split and train RF
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Save artifacts
    joblib.dump(rf, os.path.join(models_dir, "model.joblib"))
    joblib.dump(mappings, os.path.join(models_dir, "mappings.joblib"))
    joblib.dump(imputer, os.path.join(models_dir, "imputer.joblib"))
    joblib.dump(FEATURES, os.path.join(models_dir, "features.joblib"))

    # Build dropdown options (same format your app expects)
    dropdown_options = {
        "classification": sorted(df["classification"].dropna().unique().tolist()),
        "code": sorted(df["code"].dropna().unique().tolist()),
        "implanted": sorted(df["implanted"].fillna("None").unique().tolist()),
        "name_device": sorted(df["name_device"].fillna("None").unique().tolist()),
        "name_manufacturer": sorted(df["name_manufacturer"].fillna("None").unique().tolist()),
    }
    joblib.dump(dropdown_options, os.path.join(models_dir, "dropdown_options.joblib"))

    print(f"Saved model artifacts to {models_dir}")


if __name__ == "__main__":
    train_and_save()
