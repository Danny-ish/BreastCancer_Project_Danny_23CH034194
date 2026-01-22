from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model & scaler
model = joblib.load(os.path.join('model', 'breast_cancer_model.joblib'))
scaler = joblib.load(os.path.join('model', 'scaler.joblib'))

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = ""
    
    if request.method == 'POST':
        try:
            # Get 5 values from form
            radius_mean     = float(request.form['radius_mean'])
            texture_mean    = float(request.form['texture_mean'])
            perimeter_mean  = float(request.form['perimeter_mean'])
            area_mean       = float(request.form['area_mean'])
            concavity_mean  = float(request.form['concavity_mean'])

            # Make array → very important to match training order
            input_data = np.array([[
                radius_mean,
                texture_mean,
                perimeter_mean,
                area_mean,
                concavity_mean
            ]])

            # Scale
            input_scaled = scaler.transform(input_data)

            # Predict
            pred = model.predict(input_scaled)[0]
            
            if pred == 1:
                prediction = "Malignant (Cancerous)"
            else:
                prediction = "Benign (Not Cancerous)"

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)   # ← important change