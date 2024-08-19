import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
m = pd.read_pickle("fbcrypto.pkl")

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def prediction():
    if request.method == 'POST':
        ds = request.form['Date']
        ds = str(ds)
        next_day = ds
        
        future = m.make_future_dataframe(periods=365)
        forecast = m.predict(future)
        
        prediction = forecast[forecast['ds'] == next_day]['yhat'].item()
        prediction = round(prediction, 2)
        print(prediction)
        return render_template('predict.html', prediction_text="Bitcoin Price on selected date is $ {} Cents".format(prediction))
    return render_template('predict.html')

if __name__ == "__main__":
    app.run(debug=False)
