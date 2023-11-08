#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import plotly 
import plotly.express as px

import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)

from scipy import stats



# # Exploratory data Analysis

# ## Reading and being Familiar with Data

df0 = pd.read_csv('data.csv', sep = ',')
df = df0.copy()



# ## Data Preparation

# **Rename the columns**

df.columns = df.columns.str.replace('-', '_').str.lower()
df.columns


# **Checking the Categoric Columns**

df.select_dtypes('object').nunique()

for col in df.select_dtypes('object').columns:
    print(col)
    print('--'*8)
    print(df[col].value_counts(dropna=False))
    print('--'*20)


    
# ## Cleaning the data

df.isin(['?']).any()

# **Replace '?' with np.nan**

df.workclass.replace('?', np.nan, inplace=True)
df.occupation.replace('?', np.nan, inplace=True)
df.native_country.replace('?', np.nan, inplace=True)

# **Drop the duplicates**

df.drop_duplicates(keep='first', inplace=True)
df.shape

# **Check for missing values**

df.isnull().sum()

# **Dropping unnecessary columns**

drop_columns = ['education_num', 'relationship']

df.drop(drop_columns, axis=1, inplace=True)
df.shape

# **Dropping missing values**

df.dropna(inplace=True)
df.shape



# ## Bi-Multivariate Analysis

numerical= df.select_dtypes('number').columns.to_list()

categorical = df.drop('salary', axis=1).select_dtypes('object').columns.to_list()



# ### Salary (Target Feature)

print('nunique :', df.salary.nunique())
print('---'*9)
df.salary.value_counts(dropna=False)

df['salary'] = df['salary'].replace({'<=50K': 'below_50K', '>50K': 'above_50K'})
df.salary.value_counts(dropna=False)



# ### Numeric columns

# **Age**

print('Range:', np.min(df['age']), '-', np.max(df['age']))
print('---'*8)

df.age.describe()

df.groupby('salary')['age'].describe()

# **fnlwgt**

print('Range:', np.min(df['fnlwgt']), '-', np.max(df['fnlwgt']))
print('---'*8)

df.fnlwgt.describe()

df.groupby('salary')['fnlwgt'].describe()

# **capital_gain**

print('Range:', np.min(df['capital_gain']), '-', np.max(df['capital_gain']))
print('---'*8)

df.capital_gain.describe()

df.groupby('salary')['capital_gain'].describe()

# **capital_loss**

print('Range:', np.min(df['capital_loss']), '-', np.max(df['capital_loss']))
print('---'*8)

df.capital_loss.describe()

df.groupby('salary')['capital_loss'].describe()

# **hours_per_week**

print('Range:', np.min(df['hours_per_week']), '-', np.max(df['hours_per_week']))
print('---'*8)

df.hours_per_week.describe()

df.groupby('salary')['hours_per_week'].describe()



# ### Categorical columns

# **Workclass**

print('nunique :', df.workclass.nunique())
print('---'*9)

df.workclass.value_counts(dropna=False)

plt.figure(figsize=(12, 5))
ax = sns.countplot(data=df, 
                   x='workclass', 
                   order=df['workclass'].value_counts().index)
plt.title('The Distribution of Workclass', fontsize=18, color='darkblue')
plt.xticks(rotation=0)
for container in ax.containers:
    ax.bar_label(container);

df.workclass.value_counts(dropna=False)

df.workclass.value_counts(normalize=True)

workclass = df.groupby('workclass').salary.value_counts(normalize=True).reset_index(name='percentage')
workclass

# **Education**

print('nunique :', df.education.nunique())
print('---'*9)

df.education.value_counts(dropna=False)

df.education.value_counts(normalize=True)

education = df.groupby('education').salary.value_counts(normalize=True).reset_index(name='percentage')
education

def mapping_education(x):
    if x in ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th']:
        return 'low_level_grade'
    elif x in ['HS-grad', 'Some-college', 'Assoc-voc', 'Assoc-acdm']:
        return 'medium_level_grade'
    elif x in ['Bachelors', 'Masters', 'Prof-school', 'Doctorate']:
        return 'high_level_grade'

df['education'] = df.education.apply(mapping_education)

print('nunique :', df.education.nunique())
print('---'*9)

df.education.value_counts(dropna=False)

# **marital_status**

print('nunique :', df.marital_status.nunique())
print('---'*9)

df.marital_status.value_counts(dropna=False)

df.marital_status.value_counts(normalize=True)

marital_status = df.groupby('marital_status').salary.value_counts(normalize=True).reset_index(name='percentage')
marital_status

marital_status = marital_status.sort_values(by=['marital_status', 'salary'])

def mapping_marital_status(x):
    if x in ['Never-married', 'Divorced', 'Separated', 'Widowed']:
        return 'unmarried'
    elif x in ['Married-civ-spouse', 'Married-AF-spouse', 'Married-spouse-absent']:
        return 'married'

df['marital_status'] = df.marital_status.apply(mapping_marital_status)

print('nunique :', df.marital_status.nunique())
print('---'*9)

df.marital_status.value_counts(dropna=False)

# **occupation**

print('nunique :', df.occupation.nunique())
print('---'*9)

df.occupation.value_counts(dropna=False)

df.occupation.value_counts(normalize=True)

occupation = df.groupby('occupation').salary.value_counts(normalize=True).reset_index(name='percentage')
occupation

# **race**

print('nunique :', df.race.nunique())
print('---'*9)

df.race.value_counts(dropna=False)

df.race.value_counts(normalize=True)

race = df.groupby('race').salary.value_counts(normalize=True).reset_index(name='percentage')
race

race = race.sort_values(by=['race', 'salary'])

# **sex**

print('nunique :', df.sex.nunique())
print('---'*9)

df.sex.value_counts(dropna=False)

df.sex.value_counts(normalize=True)

sex = df.groupby('sex').salary.value_counts(normalize=True).reset_index(name='percentage')
sex

# **native_country**

print('nunique :', df.native_country.nunique())
print('---'*9)

df.native_country.value_counts(dropna=False)

df.native_country.value_counts(normalize=True)

native_country = df.groupby('native_country').salary.value_counts(normalize=True).reset_index(name='percentage')
native_country

NAmerica = ['Mexico', 'Canada']
MAmerica = ['El-Salvador', 'Guatemala', 'Nicaragua', 'Honduras']
SAmerica = ['Columbia', 'Peru', 'Ecuador', 'Trinadad&Tobago']
Asia = ['India', 'Iran']
EAsia = ['South', 'Japan']
SEAsia = ['Philippines', 'Vietnam', 'Taiwan', 'Cambodia', 'Thailand', 'Laos', 'Outlying-US(Guam-USVI-etc)']
Europe = ['Germany', 'Italy', 'Poland', 'Portugal', 'Greece', 'France', 'Yugoslavia', 'Hungary', 'Holand-Netherlands']
Caribbean = ['Puerto-Rico', 'Cuba', 'Jamaica', 'Dominican-Republic', 'Haiti']
GreatBritain = ['Ireland', 'England', 'Scotland']
China_HongKong = ['China', 'Hong']

def mapping_native_country(x):
    if x == 'United-States':
        return 'US'
    elif x in ['Mexico', 'Canada']:
        return 'North America'
    elif x in ['El-Salvador', 'Guatemala', 'Nicaragua', 'Honduras']:
        return 'Mid America'
    elif x in ['Columbia', 'Peru', 'Ecuador', 'Trinadad&Tobago']:
        return 'South America'
    elif x in ['India', 'Iran']:
        return 'Asia'
    elif x in ['South', 'Japan']:
        return 'Korea&Japan'
    elif x in ['Philippines', 'Vietnam', 'Taiwan', 'Cambodia', 'Thailand', 'Laos', 'Outlying-US(Guam-USVI-etc)']:
        return 'South East Asia'
    elif x in ['Germany', 'Italy', 'Poland', 'Portugal', 'Greece', 'France', 'Yugoslavia', 'Hungary', 'Holand-Netherlands']:
        return 'Europe'
    elif x in ['Puerto-Rico', 'Cuba', 'Jamaica', 'Dominican-Republic', 'Haiti']:
        return 'Caribbean'
    elif x in ['Ireland', 'England', 'Scotland']:
        return 'Great Britain'
    elif x in ['China', 'Hong']:
        return 'China'

df['native_country'] = df.native_country.apply(mapping_native_country)

print('nunique :', df.native_country.nunique())
print('---'*9)

df.native_country.value_counts(dropna=False)



# ### Dealing with Outliers

# **age**

df.age.describe()

print('Number of rows before dropping outliers:', len(df))

q1, q3 = np.percentile(df['age'],[25,75])

iqr = stats.iqr(df['age'])

fence_low  = q1-1.5*iqr
fence_high = q3+1.5*iqr

print('q1:', q1)
print('q3:', q3)
print('fence_low:', fence_low)
print('fence_high:', fence_high)

print('Potential Outliers:', len(df[(df['age'] < fence_low) | (df['age'] > fence_high)]))

df.loc[(df['age'] < fence_low) | (df['age'] > fence_high)]['age'].value_counts(dropna=False).sort_index()

# [Retirement Age in US](https://en.wikipedia.org/wiki/Retirement_age#:~:text=Pensions%20in%20the%20United%20States,born%20in%201960%20or%20later.)

df.loc[(df['age'] < 18) | (df['age'] > 70)][['age', 'hours_per_week', 'salary']].sort_values(by='age')

len(df[df.age<18]), len(df[df.age>70])

drop_index = df[(df.age < 18) | (df.age > 75)].sort_values(by='age', ascending=False).index
drop_index

df.drop(drop_index, inplace=True)
df.shape

# **hours_per_week**

df.hours_per_week.describe()

# [Working hours in US](https://clockify.me/working-hours#:~:text=Working%20hours%20in%20US,more%20than%20other%20OECD%20countries.)

len(df[df.hours_per_week<7]), len(df[df.age>84])

drop_index = df[(df.hours_per_week < 7) | (df.hours_per_week > 84)].sort_values(by='age', ascending=False).index
drop_index

df.drop(drop_index, inplace=True)
df.shape

df.reset_index(drop=True, inplace=True)

df['salary'] = df.salary.replace({'below_50K':0, 'above_50K':1})

df['salary'].value_counts()


# # Split the data
# 
# * Split your data in train/val/test sets with 60%/20%/20% distribution.
# * Use Scikit-Learn for that (the `train_test_split` function) and set the seed to `42`.
# * Make sure that the target value (`salary`) is not in your dataframe.

from sklearn.model_selection import train_test_split

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
len(df_full_train), len(df_test)

df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)
len(df_train), len(df_val), len(df_test)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.salary.values
y_val = df_val.salary.values
y_test = df_test.salary.values

del df_train['salary']
del df_val['salary']
del df_test['salary']


# ### Selecting the final model

def train(df_full_train, y_full_train):
    dicts = df_full_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_full_train = dv.fit_transform(dicts)

    model = RandomForestClassifier(class_weight='balanced', max_depth=15, n_estimators=130, random_state=42)
    model.fit(X_full_train, y_full_train)

    return dv, model

def predict(df, dv, model):
    dicts = df_test[categorical + numerical].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred

df_full_train = df_full_train.reset_index(drop=True)

y_full_train = df_full_train.salary.values

del df_full_train['salary']

dv, model = train(df_full_train, y_full_train)
y_pred = predict(df_test, dv, model)

auc = roc_auc_score(y_test, y_pred)
auc

final_accuracy = accuracy_score(y_test, y_pred>=0.5)
final_precision = precision_score(y_test, y_pred>=0.5)
final_recall = recall_score(y_test, y_pred>=0.5)
final_f1 = f1_score(y_test, y_pred>=0.5)
final_roc_auc = roc_auc_score(y_test, y_pred)

print('accuracy_score -->', final_accuracy)
print('precision_score -->', final_precision)
print('recall_score -->', final_recall)
print('f1_score -->', final_f1)
print('roc_auc_score -->', final_roc_auc)


# ### Save the model

import pickle

output_file = 'model_rf'
output_file

f_out = open(output_file, 'wb')
pickle.dump((dv, model), f_out)
f_out.close()

model_file = 'model_rf'

with open(model_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)


# ### Load the model

import pickle

model_file = 'model_rf'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

dv, model


# ### Prediction

input = {'age': 62,
          'workclass': 'Local-gov',
          'fnlwgt': 206063,
          'education': 'medium_level_grade',
          'marital_status': 'unmarried',
          'occupation': 'Other-service',
          'race': 'White',
          'sex': 'Male',
          'capital_gain': 0,
          'capital_loss': 0,
          'hours_per_week': 45,
          'native_country': 'US'}

X = dv.transform(input)

model.predict_proba(X)

model.predict_proba(X)[0,1]

prediction = model.predict(X)[0]
prediction
