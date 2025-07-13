from flask import Flask, render_template, request
import joblib
import numpy as np
from utils.preprocess import preprocess_input

app = Flask(__name__)

model = joblib.load('models/box_office_model.pkl')
encoder = joblib.load('models/genre_encoder.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    budget = float(request.form['budget'])
    genre = request.form['genre']
    input_features = preprocess_input(budget, genre, encoder)
    prediction = model.predict([input_features])[0]
    return render_template('result.html', revenue=int(prediction))

if __name__ == "__main__":
    app.run(debug=True)
