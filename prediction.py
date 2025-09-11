# prediction.py
import os
import joblib
import numpy as np
import pandas as pd

base_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(base_dir, "models")
model_path = os.path.join(models_dir, "model.joblib")
mappings_path = os.path.join(models_dir, "mappings.joblib")
imputer_path = os.path.join(models_dir, "imputer.joblib")
features_path = os.path.join(models_dir, "features.joblib")
dropdown_path = os.path.join(models_dir, "dropdown_options.joblib")

# Risk class dictionaries copied from your code
risk_class_descriptions = {
    1: "A situation where there is a reasonable chance that a product will cause serious health problems or death.",
    2: "A situation where a product may cause a temporary or reversible health problem or where there is a slight chance that it will cause serious health problems or death.",
    3: "A situation where a product is not likely to cause any health problem or injury.",
}

risk_class_suggestions = {
    1: "Immediate action is required to address the issue. Consider recalling the product or performing urgent maintenance to prevent serious outcomes.",
    2: "Monitor the situation closely and schedule maintenance to address potential issues. A temporary or reversible health problem may occur, but serious risks are low.",
    3: "Routine maintenance is sufficient. The product is not likely to cause any health problems or injury, so immediate action is not necessary.",
}

def ensure_artifacts():
    """
    Ensure model and artifacts exist. If not, call train_model.train_and_save() to create them.
    Returns loaded model, mappings, imputer, features, dropdown_options
    """
    if not os.path.exists(model_path):
        # attempt to train if artifacts missing
        try:
            from train_model import train_and_save
            print("Model artifacts not found — training model now (this may take a while)...")
            train_and_save()
        except Exception as e:
            raise RuntimeError("Model artifacts missing and automatic training failed: " + str(e))

    # load artifacts
    model = joblib.load(model_path)
    mappings = joblib.load(mappings_path)
    imputer = joblib.load(imputer_path)
    features = joblib.load(features_path)

    if os.path.exists(dropdown_path):
        dropdown_options = joblib.load(dropdown_path)
    else:
        # fallback: try to read excel
        excel_path = os.path.join(base_dir, "final_cts.xlsx")
        if os.path.exists(excel_path):
            df = pd.read_excel(excel_path)
            dropdown_options = {
                "classification": sorted(df["classification"].dropna().unique().tolist()) if "classification" in df.columns else [],
                "code": sorted(df["code"].dropna().unique().tolist()) if "code" in df.columns else [],
                "implanted": sorted(df["implanted"].fillna("None").unique().tolist()) if "implanted" in df.columns else [],
                "name_device": sorted(df["name_device"].fillna("None").unique().tolist()) if "name_device" in df.columns else [],
                "name_manufacturer": sorted(df["name_manufacturer"].fillna("None").unique().tolist()) if "name_manufacturer" in df.columns else [],
            }
            try:
                joblib.dump(dropdown_options, dropdown_path)
            except Exception:
                pass
        else:
            dropdown_options = {
                "classification": [],
                "code": [],
                "implanted": [],
                "name_device": [],
                "name_manufacturer": [],
            }

    return model, mappings, imputer, features, dropdown_options


_model, _mappings, _imputer, _features, _dropdown_options = ensure_artifacts()


def get_dropdown_options():
    """Return the saved dropdown options (used by app.py)."""
    return _dropdown_options


def predict_new_data(new_data):
    """
    Accepts either a dict (single sample) or a pandas DataFrame with 1 row.
    Returns: (predicted_class, description, suggestion)
    """
    try:
        # normalize input to DataFrame
        if isinstance(new_data, dict):
            df_in = pd.DataFrame([new_data])
        else:
            df_in = new_data.copy()

        # Ensure all expected feature columns are present
        for col in _features:
            if col not in df_in.columns:
                df_in[col] = "Unknown"

        # Keep only expected columns in the right order
        X_raw = df_in[_features].astype(str).fillna("Unknown")

        # Map categories -> ints using saved mappings. Unknown/unseen values map to the 'Unknown' index if present, otherwise 0.
        X_enc = pd.DataFrame()
        for col in _features:
            mapping = _mappings.get(col, {})
            unknown_idx = mapping.get("Unknown", 0)
            X_enc[col] = X_raw[col].map(lambda v: mapping.get(v, unknown_idx)).astype(float)

        # Impute (saved imputer expects same columns)
        X_final = pd.DataFrame(_imputer.transform(X_enc), columns=X_enc.columns)

        # Predict
        probs = _model.predict_proba(X_final)[0]
        classes = _model.classes_
        idx = int(np.argmax(probs))
        predicted_class = int(classes[idx]) if hasattr(classes[idx], "__int__") else classes[idx]
        # For user display we return description and suggestion from dictionaries
        description = risk_class_descriptions.get(predicted_class, "")
        suggestion = risk_class_suggestions.get(predicted_class, "")

        return predicted_class, description, suggestion

    except Exception as e:
        # return a useful error tuple that your app can show
        return f"Error in prediction: {str(e)}", "", ""
