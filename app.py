from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/', methods=["POST"])
def analyze():
    if request.method == 'POST':
        # Extracting features from form input
        petal_length = request.form['petal_length']
        sepal_length = request.form['sepal_length']
        petal_width = request.form['petal_width']
        sepal_width = request.form['sepal_width']

        # Data cleaning and reshaping
        sample_data = [sepal_length, sepal_width, petal_length, petal_width]
        clean_data = [float(i) for i in sample_data]
        input_data = np.array(clean_data).reshape(1, -1)

        # Load and predict with Logistic Regression model
        logit_model = joblib.load('data/logit_model_iris.pkl')
        result_prediction = logit_model.predict(input_data)

        return render_template('index.html', petal_width=petal_width,
                               sepal_width=sepal_width,
                               sepal_length=sepal_length,
                               petal_length=petal_length,
                               clean_data=clean_data,
                               result_prediction=result_prediction,
                               model_selected="Logistic Regression")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
