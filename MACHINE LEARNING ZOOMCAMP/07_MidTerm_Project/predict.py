#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

import pickle
from flask import Flask
from flask import request
from flask import jsonify

# ### Load the model

model_file = 'model_rf'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

dv, model

# ### Prediction

app = Flask('salary_estimation') # give an identity to your web service

@app.route('/predict', methods=['POST']) # use decorator to add Flask's functionality to our function
def predict():
    employee = request.get_json()
    
    X = dv.transform([employee])
    y_pred = model.predict_proba(X)[0,1]
    salary_estimation = y_pred >= 0.5
    
    result = {'salary_probability': float(y_pred),
              'salary_estimation': bool(salary_estimation)}
    return jsonify(result)

# prediction = model.predict(X)[0]
# prediction

# print('input', input)
# print('salary estimation', y_pred)

if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0', port=9696) # run the code in local machine with the debugging mode true and port 9696
