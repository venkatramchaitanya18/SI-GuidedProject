from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model
model = joblib.load('fetal_health_model.pkl')

@app.route('/home_page')
def some_page():
    return render_template('index.html')


@app.route('/')
def home():
    return render_template('login.html')
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the values from the form
    features = [
        float(request.form['baseline_value']),
        float(request.form['accelerations']),
        float(request.form['fetal_movement']),
        float(request.form['uterine_contractions']),
        float(request.form['light_decelerations']),
        float(request.form['severe_decelerations']),
        float(request.form['prolongued_decelerations']),
        float(request.form['abnormal_short_term_variability']),
        float(request.form['mean_value_of_short_term_variability']),
        float(request.form['percentage_of_time_with_abnormal_long_term_variability']),
        float(request.form['mean_value_of_long_term_variability']),
        float(request.form['histogram_width']),
        float(request.form['histogram_min']),
        float(request.form['histogram_max']),
        float(request.form['histogram_number_of_peaks']),
        float(request.form['histogram_number_of_zeroes']),
        float(request.form['histogram_mode']),
        float(request.form['histogram_mean']),
        float(request.form['histogram_median']),
        float(request.form['histogram_variance']),
        float(request.form['histogram_tendency'])
    ]

    # Convert the features into a numpy array
    features = np.array([features])

    # Make the prediction
    prediction = model.predict(features)

    # Convert the prediction to a human-readable label
    if prediction == 0:
        result = "Normal"
    elif prediction == 1:
        result = "Suspect"
    elif prediction == 2:
        result = "Pathological"

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
