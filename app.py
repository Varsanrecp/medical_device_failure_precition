from flask import Flask, render_template, request
import pandas as pd
from prediction import predict_new_data  # Ensure this matches the function from prediction.py

app = Flask(__name__)

# Load the dataset and extract unique values for dropdowns
df = pd.read_excel(r'C:\Users\91978\Desktop\medical_device_failure_prediction-main\final_cts.xlsx')

# Extract unique values for each relevant column (after removing unwanted fields)
dropdown_options = {
    'classification': sorted(df['classification'].dropna().unique().tolist()),
    'code': sorted(df['code'].dropna().unique().tolist()),
    'implanted': sorted(df['implanted'].fillna('None').unique().tolist()),
    'name_device': sorted(df['name_device'].fillna('None').unique().tolist()),
    'name_manufacturer': sorted(df['name_manufacturer'].fillna('None').unique().tolist()),
}

@app.route('/')
def index():
    return render_template('index.html', dropdown_options=dropdown_options)

@app.route('/predict', methods=['POST'])
def predict():
    # Extract form data
    form_data = {
        'classification': request.form.get('classification'),
        'code': request.form.get('code'),
        'implanted': request.form.get('implanted'),
        'name_device': request.form.get('name_device'),
        'name_manufacturer': request.form.get('name_manufacturer'),
    }

    # Convert form data into a DataFrame for prediction
    new_data = pd.DataFrame([form_data])
    
    # Make prediction
    prediction_result = predict_new_data(new_data)
    predicted_class, description, suggestion = prediction_result  # Assuming predict_new_data returns a tuple

    # Display the results on the page
    return render_template('result.html', predicted_class=predicted_class, description=description, suggestion=suggestion)

if __name__ == '__main__':
    app.run(debug=True)
