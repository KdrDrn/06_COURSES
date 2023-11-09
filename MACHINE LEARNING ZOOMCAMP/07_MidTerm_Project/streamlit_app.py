import streamlit as st
import pickle
import pandas as pd

st.sidebar.title('Salary Estimation')
html_temp = """
<div style="background-color:blue;padding:10px">
<h2 style="color:white;text-align:center;">Machine Learning Zoomcamp </h2>
</div>"""
st.markdown(html_temp, unsafe_allow_html=True)


age=st.sidebar.slider("Employee Age:", 15, 100, step=1)
native_country=st.sidebar.selectbox("Employee Native Country", ('US', 'North America', 'Caribbean', 'Europe', 'South East Asia', 'Mid  America', 'Asia', 'South America', 'Korea&Japan', 'Great Britain', 'China'))
race=st.sidebar.selectbox("Employee Race", ('White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'))
sex=st.sidebar.selectbox('Employee Gender',('Male','Female'))
marital_status=st.sidebar.radio('Employee Marital_Status',('unmarried','married'))
education=st.sidebar.selectbox("Employee Education", ('medium_level_grade', 'high_level_grade', 'low_level_grade'))
workclass=st.sidebar.selectbox("Employee Workclass", ('Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov', 'Self-emp-inc', 'Federal-gov', 'Without-pay'))
occupation=st.sidebar.selectbox("Employee Occupation", ('Prof-specialty', 'Craft-repair', 'Exec-managerial', 'Adm-clerical', 'Sales', 'Other-service', 'Machine-op-inspct', 'Transport-moving', 'Handlers-cleaners', 'Farming-fishing', 'Tech-support', 'Protective-serv', 'Priv-house-serv', 'Armed-Forces'))
fnlwgt=st.sidebar.slider("Employee  fnlwgt", 13500,1485000, step=1000)
capital_gain=st.sidebar.slider("What capital_gain", 0,100000, step=50)
capital_loss=st.sidebar.slider("Employee  capital_loss", 0,4500, step=25)
hours_per_week=st.sidebar.slider("Employee  hours_per_week", 0,100, step=1)


# ### Load the model

import pickle

model_file = 'model_rf'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

my_dict = {
    "age": age,
    "workclass": workclass,
    "fnlwgt": fnlwgt,
    'education':education,
    "marital_status": marital_status,
    "occupation": occupation,
    "race": race,
    "sex": sex,
    'capital_gain':capital_gain,
    "capital_loss": capital_loss,
    'hours_per_week':hours_per_week,
    "native_country": native_country
    
}

df = pd.DataFrame.from_dict([my_dict])


st.header("The Infos of the Employee")
st.table(df)


X = dv.transform(my_dict)

st.subheader("Press predict if configuration is okay")

if st.button("Predict"):
    y_pred = model.predict_proba(X)[0,1]
    salary_estimation = y_pred >= 0.5
    if salary_estimation==1:
        st.success("The salary_estimation above 50K")
    else:
        st.warning("The salary_estimation below 50K")
