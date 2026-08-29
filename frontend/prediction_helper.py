import joblib
import pandas as pd
from pathlib import Path

# -------------------------------------------------------------------------
# 1. Dynamic Path Resolution (Works locally on Windows & on Streamlit Cloud)
# -------------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
MODELS_DIR = PROJECT_ROOT / "research" / "models"

# Load models and column preprocessors
model_young = joblib.load(MODELS_DIR / "model_young.joblib")
model_rest = joblib.load(MODELS_DIR / "model_rest.joblib")
preprocessor_young = joblib.load(MODELS_DIR / "preprocessor_young.joblib")
preprocessor_rest = joblib.load(MODELS_DIR / "preprocessor_rest.joblib")


# -------------------------------------------------------------------------
# 2. Risk Score Calculation Logic
# -------------------------------------------------------------------------
def calculate_total_risk_score(medical_history: str) -> int:
    """Calculates total risk score based on medical history conditions."""
    risk_scores = {
        "diabetes": 6,
        "heart disease": 8,
        "high blood pressure": 6,
        "thyroid": 5,
        "no disease": 0,
        "none": 0
    }

    if not medical_history:
        return 0

    diseases = str(medical_history).lower().split(" & ")
    return sum(risk_scores.get(d.strip(), 0) for d in diseases)


# -------------------------------------------------------------------------
# 3. Dynamic Preprocessing & Feature Alignment
# -------------------------------------------------------------------------
def preprocess_input(input_dict: dict) -> pd.DataFrame:
    """Converts user input dictionary to a DataFrame and runs it through

    the fitted ColumnTransformer without missing column errors.
    """
    age = input_dict.get('Age', 30)
    medical_history = input_dict.get('Medical History', 'none')

    # Select preprocessor based on age segment
    preprocessor = preprocessor_young if age <= 25 else preprocessor_rest

    # Base feature dictionary mapped from UI keys
    raw_record = {
        'age': age,
        'gender': input_dict.get('Gender', 'Male'),
        'region': input_dict.get('Region', 'Northwest'),
        'marital_status': input_dict.get('Marital Status', 'Unmarried'),
        'bmi_category': input_dict.get('BMI Category', 'Normal'),
        'smoking_status': input_dict.get('Smoking Status', 'No'),
        'employment_status': input_dict.get('Employment Status', 'Salaried'),
        'insurance_plan': input_dict.get('Insurance Plan', 'Bronze'),
        'income_level': input_dict.get('Income Level', '<10L'),
        'number_of_dependants': input_dict.get('Number of Dependants', 0),
        'income_lakhs': input_dict.get('Income in Lakhs', 10),
        'total_risk_score': calculate_total_risk_score(medical_history),
        'genetical_risk': input_dict.get('Genetical Risk', 0),
        'medical_history': medical_history,
        'disease1': 'none',
        'disease2': 'none'
    }

    raw_df = pd.DataFrame([raw_record])

    # Dynamically inject and align all columns expected by the fitted ColumnTransformer
    if hasattr(preprocessor, "feature_names_in_"):
        for col in preprocessor.feature_names_in_:
            if col not in raw_df.columns:
                raw_df[col] = 0
        raw_df = raw_df[list(preprocessor.feature_names_in_)]

    # Transform through the pipeline
    transformed = preprocessor.transform(raw_df)

    # Convert to DataFrame if output is a NumPy array
    if not isinstance(transformed, pd.DataFrame):
        try:
            feature_names = preprocessor.get_feature_names_out()
        except Exception:
            feature_names = None
        transformed = pd.DataFrame(transformed, columns=feature_names)

    # Clean up redundant features if they were dropped prior to model fitting
    if 'income_level' in transformed.columns:
        transformed = transformed.drop(columns=['income_level'])

    return transformed


# -------------------------------------------------------------------------
# 4. Premium Prediction
# -------------------------------------------------------------------------
def predict(input_dict: dict) -> int:
    """Predicts the annual health insurance premium."""
    processed_df = preprocess_input(input_dict)
    age = input_dict.get('Age', 30)

    if age <= 25:
        prediction = model_young.predict(processed_df)
    else:
        prediction = model_rest.predict(processed_df)

    return max(0, int(prediction[0]))