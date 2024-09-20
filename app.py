from flask import Flask, request, render_template, flash
import pandas as pd
import pickle
from sklearn.metrics import classification_report, confusion_matrix
import os

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Load the model
model_path = "naadi_diagnostic_model_SMOT.pkl"
model = None

if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
else:
    print("Model file not found.")

# Home route
@app.route('/', methods=['GET', 'POST'])
def index():
    diagnosis_str = None
    classification_report_str = None
    confusion_matrix_str = None

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'error')
            return render_template('index.html')

        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'error')
            return render_template('index.html')

        if file:
            try:
                input_data = pd.read_csv(file)

                required_columns = ['Vata_pressure', 'Pitta_pressure', 'Kapha_pressure', 'Oxygen_level']
                if not all(col in input_data.columns for col in required_columns):
                    flash(f'CSV must contain these columns: {", ".join(required_columns)}', 'error')
                    return render_template('index.html')

                input_data = input_data[required_columns]

                if model is None:
                    flash('Model not loaded. Please check the model file.', 'error')
                    return render_template('index.html')

                if not input_data.empty:
                    predictions = model.predict(input_data)
                    unique_predictions = set(predictions.astype(str))
                    diagnosis_str = ', '.join(unique_predictions)

                    # Assuming you have a way to retrieve y_test (true labels) for evaluation
                    # Here, just an example of how to create a mock y_test based on the input size
                    y_test = ['gastric'] * (len(input_data) // 2) + ['migraine'] * (len(input_data) - (len(input_data) // 2))  # Mock labels
                    y_test = y_test[:len(input_data)]  # Ensure size matches

                    classification_report_str = classification_report(y_test, predictions, output_dict=True)
                    confusion_matrix_str = confusion_matrix(y_test, predictions).tolist()

                    flash('Diagnosis successful!', 'success')
                else:
                    flash('Input data is empty. Please check the CSV file.', 'error')

            except Exception as e:
                flash(f'Error processing file: {str(e)}', 'error')

    return render_template('index.html', diagnosis=diagnosis_str, classification_report=classification_report_str, confusion_matrix=confusion_matrix_str)

if __name__ == '__main__':
    app.run(debug=True)
