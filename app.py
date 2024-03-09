from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

filename = 'model.pkl'
classifier = pickle.load(open(filename, 'rb'))

# Define the valid ranges for each feature
feature_ranges = {
    'age': (40, 95),
    'anaemia': (0, 1),
    'CPK': (23, 7861),
    'Diabetes': (0, 1),
    'EF': (14, 80),
    'bloodpressure': (0, 1),
    'platelets': (25100, 850000),
    'SC': (0.5, 9.4),
    'SS': (110, 150),
    'Gender': (0, 1),
    'Smoking': (0, 1),
    'time': (4, 285)
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Validate input values
        input_values = {}
        error_message = None

        for key in feature_ranges:
            value = request.form[key]
            try:
                value = float(value)
                if feature_ranges[key][0] <= value <= feature_ranges[key][1]:
                    input_values[key] = value
                else:
                    error_message = f"Invalid value for {key}. Must be between {feature_ranges[key][0]} and {feature_ranges[key][1]}."
            except ValueError:
                error_message = f"{key} must be a numeric value."

        if error_message:
            return render_template('index.html', error=error_message)
        
        data = np.array([[input_values['age'], input_values['anaemia'], input_values['CPK'], input_values['Diabetes'],
                          input_values['EF'], input_values['bloodpressure'], input_values['platelets'],
                          input_values['SC'], input_values['SS'], input_values['Gender'],
                          input_values['Smoking'], input_values['time']]])

        my_prediction = classifier.predict(data)

        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
    app.run(debug=True)
