import pandas as pd
import joblib

# Load models and preprocessors
model_young = joblib.load(
    r"D:\AI-engineer Study\Projects\Project 7 Premium Prediction ver1\research\models\model_young.joblib")
model_rest = joblib.load(
    r"D:\AI-engineer Study\Projects\Project 7 Premium Prediction ver1\research\models\model_rest.joblib")
preprocessor_young = joblib.load(
    r"D:\AI-engineer Study\Projects\Project 7 Premium Prediction ver1\research\models\preprocessor_young.joblib")
preprocessor_rest = joblib.load(
    r"D:\AI-engineer Study\Projects\Project 7 Premium Prediction ver1\research\models\preprocessor_rest.joblib")


def calculate_total_risk_score(medical_history):
    """Calculates the raw total risk score based on medical history."""
    risk_scores = {
        "diabetes": 6,
        "heart disease": 8,
        "high blood pressure": 6,
        "thyroid": 5,
        "no disease": 0,
        "none": 0
    }

    # Split composite string and sum scores
    diseases = str(medical_history).lower().split(" & ")
    return sum(risk_scores.get(disease.strip(), 0) for disease in diseases)


def preprocess_input(input_dict):
    """Formats raw user input and processes it through the saved ColumnTransformer."""
    # 1. Map UI input dictionary keys to the column names expected by the ColumnTransformer
    raw_record = {
        'age': input_dict['Age'],
        'gender': input_dict['Gender'],
        'region': input_dict['Region'],
        'marital_status': input_dict['Marital Status'],
        'bmi_category': input_dict['BMI Category'],
        'smoking_status': input_dict['Smoking Status'],
        'employment_status': input_dict['Employment Status'],
        'insurance_plan': input_dict['Insurance Plan'],
        'income_level': input_dict.get('Income Level', '<10L'),  # Placeholder if preprocessor requires column
        'number_of_dependants': input_dict['Number of Dependants'],
        'income_lakhs': input_dict['Income in Lakhs'],
        'total_risk_score': calculate_total_risk_score(input_dict['Medical History'])
    }

    # Convert single input dictionary into a 1-row DataFrame
    raw_df = pd.DataFrame([raw_record])

    # 2. Select the appropriate preprocessor based on age
    preprocessor = preprocessor_young if input_dict['Age'] <= 25 else preprocessor_rest

    # 3. Transform data using the fitted ColumnTransformer
    transformed = preprocessor.transform(raw_df)

    # Convert back to DataFrame if transformer outputs a numpy array
    if not isinstance(transformed, pd.DataFrame):
        feature_names = preprocessor.get_feature_names_out()
        transformed = pd.DataFrame(transformed, columns=feature_names)

    # 4. Drop income_level if it was dropped prior to model training
    if 'income_level' in transformed.columns:
        transformed = transformed.drop(columns=['income_level'])

    return transformed


def predict(input_dict):
    """Generates the insurance premium prediction."""
    processed_df = preprocess_input(input_dict)

    if input_dict['Age'] <= 25:
        prediction = model_young.predict(processed_df)
    else:
        prediction = model_rest.predict(processed_df)

    return int(prediction[0])