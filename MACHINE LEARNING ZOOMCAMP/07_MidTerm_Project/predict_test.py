#!/usr/bin/env python
# coding: utf-8

import requests


# In[19]:


url = 'http://localhost:9696/predict'

employee_1 = {
          "age": 43,
          "workclass": "Local-gov",
          "fnlwgt": 206063,
          "education": "medium_level_grade",
          "marital_status": "unmarried",
          "occupation": "Other-service",
          "race": "White",
          "sex": "Male",
          "capital_gain": 0,
          "capital_loss": 0,
          "hours_per_week": 45,
          "native_country": "US"
           }

response = requests.post(url, json=employee_1).json()
print(response)


if response['salary_estimation'] == True:
    print("success")
else:
    print("failure")

# employee_2 = {"age": 43,
#              "workclass": "Private",
#              "fnlwgt": 213844,
#              "education": "HS-medium_level_grade",
#              "marital-status": "married",
#              "occupation": "Craft-repair",
#              "race": "Black",
#              "sex": "Male",
#              "capital-gain": 0,
#              "capital-loss": 0,
#              "hours-per-week": 42,
#              "native-country": "US"
#              }


# response = requests.post(url, json=employee_2).json()
# response


# if response['salary_estimation'] == True:
#     print("success")

