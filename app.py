from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model/model.pkl")
encoder = joblib.load("model/scaler.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    location = request.form['location']
    area = float(request.form['area'])
    bedrooms = int(request.form['bedrooms'])
    bathrooms = int(request.form['bathrooms'])
    floors = int(request.form['floors'])
    parking = int(request.form['parking'])
    age = int(request.form['age'])

    location_encoded = encoder.transform([location])[0]

    features = np.array([[

        location_encoded,
        area,
        bedrooms,
        bathrooms,
        floors,
        parking,
        age

    ]])

    prediction = model.predict(features)[0]

    if prediction < 5000000:
        category = "Budget House"
    elif prediction < 10000000:
        category = "Premium House"
    else:
        category = "Luxury House"

    return render_template(
        "result.html",
        price=f"{prediction:,.0f}",
        category=category
    )

if __name__ == '__main__':
    app.run(debug=True)