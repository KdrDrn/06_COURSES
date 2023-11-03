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

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mutual_info_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.feature_extraction import DictVectorizer

from sklearn.linear_model import LogisticRegression

from tqdm.auto import tqdm

from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text

from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb

import pickle

# User Defined Function

def show_distribution(col):
    '''
    This function will make a distribution (graph) and display it    '''

    # Get statistics
    min_val = col.min()
    max_val = col.max()
    mean_val = col.mean()
    med_val = col.median()
    mod_val = col.mode()[0]
    print('Minimum:{:.2f}\nMean:{:.2f}\nMedian:{:.2f}\nMode:{:.2f}\nMaximum:{:.2f}\n'.format(min_val,
                                                                                            mean_val,
                                                                                            med_val,
                                                                                            mod_val,
                                                                                            max_val))
    # Create a figure for 3 subplots (3 rows, 1 column)
    fig, ax = plt.subplots(3, 1, figsize = (15,15))
    # Plot the histogram   
    sns.histplot(col, ax=ax[0])
    plt.ylabel('Frequency', fontsize=10)
    # Plot the boxplot   
    sns.boxplot(col, orient='h', ax=ax[1])
    plt.xlabel('value', fontsize=10)    
    # Plot density
    sns.kdeplot(col, ax=ax[2])
    # Show the mean, median, and mode
    plt.axvline(x=min_val, color = 'gray', linestyle='dashed', linewidth = 2, label='Minimum')
    plt.axvline(x=mean_val, color = 'cyan', linestyle='dashed', linewidth = 2, label='Mean')
    plt.axvline(x=med_val, color = 'red', linestyle='dashed', linewidth = 2, label='Median')
    plt.axvline(x=mod_val, color = 'yellow', linestyle='dashed', linewidth = 2, label='Mode')
    plt.axvline(x=max_val, color = 'gray', linestyle='dashed', linewidth = 2, label='Maximum')
    plt.legend(loc='upper right')
    plt.legend()    
    # Add a title to the Figure
    fig.suptitle('Data Distribution', fontsize=15)

# # Exploratory data Analysis

# ## Reading and being Familiar with Data

df0 = pd.read_csv('data.csv', sep = ',')
df = df0.copy()

df.head()

df.sample(5)

df.tail(5)

df.shape

df.info()

df.describe()

df.describe(include='object')

df.isnull().sum()

round(df.isnull().sum()/df.shape[0]*100, 2)

df.duplicated().sum()

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

df[['education', 'education_num']].isnull().sum()

pd.crosstab(df.education, df.education_num)

df[['marital_status', 'relationship']].isnull().sum()

pd.crosstab(df.marital_status, df.relationship)

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

print(numerical)
print()
print(categorical)

# ### Salary (Target Feature)

print('nunique :', df.salary.nunique())
print('---'*9)
df.salary.value_counts(dropna=False)

df['salary'] = df['salary'].replace({'<=50K': 'below_50K', '>50K': 'above_50K'})
df.salary.value_counts(dropna=False)

# Visualizing the number of people in each category of 'salary'

fig, ax = plt.subplots(figsize=(10, 5))
sns.countplot(data=df, 
              x='salary', 
              )
ax.set_title('Total Number of People by Income Level', fontsize=18)
ax.bar_label(ax.containers[0], color='darkred', size=12);
fig, ax = plt.subplots(figsize=(6, 6))
ax.pie(x=df.salary.value_counts().values, 
       labels=['<=50K', '>50K'], 
       autopct='%.1f%%',
       explode=(0, 0.1),
       textprops={'fontsize': 12}
       )
plt.title('Percentage of Income-Levels', fontdict = {'fontsize': 14})
plt.show()

# ### Numeric columns

numerical

# **Age**

print('Range:', np.min(df['age']), '-', np.max(df['age']))
print('---'*8)

df.age.describe()

show_distribution(df['age'])

df.groupby('salary')['age'].describe()

pd.crosstab(df['age'], df['salary']).iplot(kind='bar', title = 'age and salary')

# **fnlwgt**

print('Range:', np.min(df['fnlwgt']), '-', np.max(df['fnlwgt']))
print('---'*8)

df.fnlwgt.describe()

show_distribution(df['fnlwgt'])

df.groupby('salary')['fnlwgt'].describe()

pd.crosstab(df['fnlwgt'], df['salary']).iplot(kind='bar', title = 'fnlwgt and salary')

# **capital_gain**

print('Range:', np.min(df['capital_gain']), '-', np.max(df['capital_gain']))
print('---'*8)

df.capital_gain.describe()

show_distribution(df['capital_gain'])

df.groupby('salary')['capital_gain'].describe()

pd.crosstab(df['capital_gain'], df['salary']).iplot(kind='bar', title = 'capital_gain and salary')

# **capital_loss**

print('Range:', np.min(df['capital_loss']), '-', np.max(df['capital_loss']))
print('---'*8)

df.capital_loss.describe()

show_distribution(df['capital_loss'])

df.groupby('salary')['capital_loss'].describe()

pd.crosstab(df['capital_loss'], df['salary']).iplot(kind='bar', title = 'capital_loss and salary')

# **hours_per_week**

print('Range:', np.min(df['hours_per_week']), '-', np.max(df['hours_per_week']))
print('---'*8)

df.hours_per_week.describe()

show_distribution(df['hours_per_week'])

df.groupby('salary')['hours_per_week'].describe()

pd.crosstab(df['hours_per_week'], df['salary']).iplot(kind='bar', title = 'hours_per_week and salary')

# ### Categorical columns

categorical

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

df.workclass.value_counts(normalize=True)

plt.figure(figsize=(12, 5))
ax = sns.barplot(data=df, 
                 x=round(df['workclass'].value_counts(normalize=True),2).values, 
                 y=df['workclass'].value_counts(normalize=True).index)
plt.title('The Distribution of Workclass', fontsize=18, color='darkblue')
plt.xticks(rotation=0)
for container in ax.containers:
    ax.bar_label(container);

pd.crosstab(df['workclass'], df['salary']).iplot(kind='bar', title = 'workclass and salary')

workclass = df.groupby('workclass').salary.value_counts(normalize=True).reset_index(name='percentage')
workclass

workclass = workclass.sort_values(by=['workclass', 'salary'])

fig, ax = plt.subplots(figsize=(12, 6))
ax = sns.barplot(data=workclass, 
                 x='workclass', 
                 y='percentage', 
                 hue='salary', 
                 order=workclass.groupby('workclass').percentage.sum().sort_values(ascending=False).index)
plt.title('The Distribution of Workclass by Salary', fontsize=18, color='darkblue')
plt.xticks(rotation=60)
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f');

# **Education**

print('nunique :', df.education.nunique())
print('---'*9)

df.education.value_counts(dropna=False)

plt.figure(figsize=(12, 5))
ax = sns.countplot(data=df, 
                   x='education', 
                   order=df['education'].value_counts().index)
plt.title('The Distribution of Education', fontsize=18, color='darkblue')
plt.xticks(rotation=90)
for container in ax.containers:
    ax.bar_label(container);

df.education.value_counts(normalize=True)

plt.figure(figsize=(12, 5))
ax = sns.barplot(data=df, 
                 x=round(df['education'].value_counts(normalize=True),2).values, 
                 y=df['education'].value_counts(normalize=True).index)
plt.title('The Distribution of Education', fontsize=18, color='darkblue')
plt.xticks(rotation=0)
for container in ax.containers:
    ax.bar_label(container);

pd.crosstab(df['education'], df['salary']).iplot(kind='bar', title = 'education and salary')

education = df.groupby('education').salary.value_counts(normalize=True).reset_index(name='percentage')
education

education = education.sort_values(by=['education', 'salary'])

fig, ax = plt.subplots(figsize=(12, 6))
ax = sns.barplot(data=education, 
                 x='education', 
                 y='percentage', 
                 hue='salary', 
                 order=education.groupby('education').percentage.sum().sort_values(ascending=False).index)
plt.title('The Distribution of Education by Salary', fontsize=18, color='darkblue')
plt.xticks(rotation=60)
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f');

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

pd.crosstab(df['education'], df['salary']).iplot(kind='bar', title = 'education and salary')

# **marital_status**

print('nunique :', df.marital_status.nunique())
print('---'*9)

df.marital_status.value_counts(dropna=False)

plt.figure(figsize=(12, 5))
ax = sns.countplot(data=df, 
                   x='marital_status', 
                   order=df['marital_status'].value_counts().index)
plt.title('The Distribution of Marital_Status', fontsize=18, color='darkblue')
plt.xticks(rotation=90)
for container in ax.containers:
    ax.bar_label(container);

df.marital_status.value_counts(normalize=True)

plt.figure(figsize=(12, 5))
ax = sns.barplot(data=df, 
                 x=round(df['marital_status'].value_counts(normalize=True),2).values, 
                 y=df['marital_status'].value_counts(normalize=True).index)
plt.title('The Distribution of Marital_Status', fontsize=18, color='darkblue')
plt.xticks(rotation=0)
for container in ax.containers:
    ax.bar_label(container);

pd.crosstab(df['marital_status'], df['salary']).iplot(kind='bar', title = 'marital_status and salary')

marital_status = df.groupby('marital_status').salary.value_counts(normalize=True).reset_index(name='percentage')
marital_status

marital_status = marital_status.sort_values(by=['marital_status', 'salary'])

fig, ax = plt.subplots(figsize=(12, 6))
ax = sns.barplot(data=marital_status, 
                 x='marital_status', 
                 y='percentage', 
                 hue='salary', 
                 order=marital_status.groupby('marital_status').percentage.sum().sort_values(ascending=False).index)
plt.title('The Distribution of Marital_Status by Salary', fontsize=18, color='darkblue')
plt.xticks(rotation=60)
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f');

def mapping_marital_status(x):
    if x in ['Never-married', 'Divorced', 'Separated', 'Widowed']:
        return 'unmarried'
    elif x in ['Married-civ-spouse', 'Married-AF-spouse', 'Married-spouse-absent']:
        return 'married'

df['marital_status'] = df.marital_status.apply(mapping_marital_status)

print('nunique :', df.marital_status.nunique())
print('---'*9)

df.marital_status.value_counts(dropna=False)

pd.crosstab(df['marital_status'], df['salary']).iplot(kind='bar', title = 'marital_status and salary')

# **occupation**

print('nunique :', df.occupation.nunique())
print('---'*9)

df.occupation.value_counts(dropna=False)

plt.figure(figsize=(12, 5))
ax = sns.countplot(data=df, 
                   x='occupation', 
                   order=df['occupation'].value_counts().index)
plt.title('The Distribution of Occupation', fontsize=18, color='darkblue')
plt.xticks(rotation=90)
for container in ax.containers:
    ax.bar_label(container);

df.occupation.value_counts(normalize=True)

plt.figure(figsize=(12, 5))
ax = sns.barplot(data=df, 
                 x=round(df['occupation'].value_counts(normalize=True),2).values, 
                 y=df['occupation'].value_counts(normalize=True).index)
plt.title('The Distribution of Occupation', fontsize=18, color='darkblue')
plt.xticks(rotation=0)
for container in ax.containers:
    ax.bar_label(container);

pd.crosstab(df['occupation'], df['salary']).iplot(kind='bar', title = 'occupation and salary')

occupation = df.groupby('occupation').salary.value_counts(normalize=True).reset_index(name='percentage')
occupation

occupation = occupation.sort_values(by=['occupation', 'salary'])

fig, ax = plt.subplots(figsize=(12, 6))
ax = sns.barplot(data=occupation, 
                 x='occupation', 
                 y='percentage', 
                 hue='salary', 
                 order=occupation.groupby('occupation').percentage.sum().sort_values(ascending=False).index)
plt.title('The Distribution of Occupation by Salary', fontsize=18, color='darkblue')
plt.xticks(rotation=60)
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f');

# **race**

print('nunique :', df.race.nunique())
print('---'*9)

df.race.value_counts(dropna=False)

plt.figure(figsize=(12, 5))
ax = sns.countplot(data=df, 
                   x='race', 
                   order=df['race'].value_counts().index)
plt.title('The Distribution of Race', fontsize=18, color='darkblue')
plt.xticks(rotation=90)
for container in ax.containers:
    ax.bar_label(container);

df.race.value_counts(normalize=True)

plt.figure(figsize=(12, 5))
ax = sns.barplot(data=df, 
                 x=round(df['race'].value_counts(normalize=True),2).values, 
                 y=df['race'].value_counts(normalize=True).index)
plt.title('The Distribution of Race', fontsize=18, color='darkblue')
plt.xticks(rotation=0)
for container in ax.containers:
    ax.bar_label(container);

pd.crosstab(df['race'], df['salary']).iplot(kind='bar', title = 'race and salary')

race = df.groupby('race').salary.value_counts(normalize=True).reset_index(name='percentage')
race

race = race.sort_values(by=['race', 'salary'])

fig, ax = plt.subplots(figsize=(12, 6))
ax = sns.barplot(data=race, 
                 x='race', 
                 y='percentage', 
                 hue='salary', 
                 order=race.groupby('race').percentage.sum().sort_values(ascending=False).index)
plt.title('The Distribution of Race by Salary', fontsize=18, color='darkblue')
plt.xticks(rotation=60)
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f');

# **sex**

print('nunique :', df.sex.nunique())
print('---'*9)

df.sex.value_counts(dropna=False)

plt.figure(figsize=(12, 5))
ax = sns.countplot(data=df, 
                   x='sex', 
                   order=df['sex'].value_counts().index)
plt.title('The Distribution of Sex', fontsize=18, color='darkblue')
plt.xticks(rotation=90)
for container in ax.containers:
    ax.bar_label(container);

df.sex.value_counts(normalize=True)

plt.figure(figsize=(12, 5))
ax = sns.barplot(data=df, 
                 x=round(df['sex'].value_counts(normalize=True),2).values, 
                 y=df['sex'].value_counts(normalize=True).index)
plt.title('The Distribution of Sex', fontsize=18, color='darkblue')
plt.xticks(rotation=0)
for container in ax.containers:
    ax.bar_label(container);

pd.crosstab(df['sex'], df['salary']).iplot(kind='bar', title = 'sex and salary')

sex = df.groupby('sex').salary.value_counts(normalize=True).reset_index(name='percentage')
sex

sex = sex.sort_values(by=['sex', 'salary'])

fig, ax = plt.subplots(figsize=(12, 6))
ax = sns.barplot(data=sex, 
                 x='sex', 
                 y='percentage', 
                 hue='salary', 
                 order=sex.groupby('sex').percentage.sum().sort_values(ascending=False).index)
plt.title('The Distribution of Sex by Salary', fontsize=18, color='darkblue')
plt.xticks(rotation=60)
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f');

# **native_country**

print('nunique :', df.native_country.nunique())
print('---'*9)

df.native_country.value_counts(dropna=False)

plt.figure(figsize=(12, 5))
ax = sns.countplot(data=df, 
                   x='native_country', 
                   order=df['native_country'].value_counts().index)
plt.title('The Distribution of Native_Country', fontsize=18, color='darkblue')
plt.xticks(rotation=90)
for container in ax.containers:
    ax.bar_label(container);

df.native_country.value_counts(normalize=True)

plt.figure(figsize=(12, 5))
ax = sns.barplot(data=df, 
                 x=round(df['native_country'].value_counts(normalize=True),2).values, 
                 y=df['native_country'].value_counts(normalize=True).index)
plt.title('The Distribution of Native_Country', fontsize=18, color='darkblue')
plt.xticks(rotation=0)
for container in ax.containers:
    ax.bar_label(container);

pd.crosstab(df['native_country'], df['salary']).iplot(kind='bar', title = 'native_country and salary')

native_country = df.groupby('native_country').salary.value_counts(normalize=True).reset_index(name='percentage')
native_country

native_country = native_country.sort_values(by=['native_country', 'salary'])

fig, ax = plt.subplots(figsize=(12, 6))
ax = sns.barplot(data=native_country, 
                 x='native_country', 
                 y='percentage', 
                 hue='salary', 
                 order=native_country.groupby('native_country').percentage.sum().sort_values(ascending=False).index)
plt.title('The Distribution of Native_Country by Salary', fontsize=18, color='darkblue')
plt.xticks(rotation=60)
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f');

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

pd.crosstab(df['native_country'], df['salary']).iplot(kind='bar', title = 'native_country and salary')

# ### Dealing with Outliers

numerical

# **age**

df.age.describe()

plt.figure(figsize=(20, 6))
plt.subplot(121)
plt.hist(df.age, bins=20)
plt.subplot(122)
plt.boxplot(df.age, whis=1.5)
plt.show()

df.age.value_counts(dropna=False).sort_index()[:20]

df.age.value_counts(dropna=False).sort_index()[-20:]

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

plt.figure(figsize=(20, 6))
plt.subplot(121)
plt.hist(df.age, bins=20)
plt.subplot(122)
plt.boxplot(df.age, whis=1.5)
plt.show()

# **hours_per_week**

df.age.describe()

plt.figure(figsize=(20, 6))
plt.subplot(121)
plt.hist(df.hours_per_week, bins=20)
plt.subplot(122)
plt.boxplot(df.hours_per_week, whis=1.5)
plt.show()

df.hours_per_week.value_counts(dropna=False).sort_index()[:20]

df.hours_per_week.value_counts(dropna=False).sort_index()[-20:]

# [Working hours in US](https://clockify.me/working-hours#:~:text=Working%20hours%20in%20US,more%20than%20other%20OECD%20countries.)

len(df[df.hours_per_week<7]), len(df[df.age>84])

drop_index = df[(df.hours_per_week < 7) | (df.hours_per_week > 84)].sort_values(by='age', ascending=False).index
drop_index

df.drop(drop_index, inplace=True)
df.shape

plt.figure(figsize=(20, 6))
plt.subplot(121)
plt.hist(df.hours_per_week, bins=20)
plt.subplot(122)
plt.boxplot(df.hours_per_week, whis=1.5)
plt.show()

df.reset_index(drop=True, inplace=True)
df.head()

df.shape

df.info()

df['salary'].value_counts()

df['salary'] = df.salary.replace({'below_50K':0, 'above_50K':1})

df['salary'].value_counts()

# # Split the data
# 
# * Split your data in train/val/test sets with 60%/20%/20% distribution.
# * Use Scikit-Learn for that (the `train_test_split` function) and set the seed to `42`.
# * Make sure that the target value (`salary`) is not in your dataframe.

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

# ## Feature importance: Mutual information
# 
# Mutual information - concept from information theory, it tells us how much 
# we can learn about one variable if we know the value of another
# 
# * https://en.wikipedia.org/wiki/Mutual_information

def mutual_info_salary_score(series):
    return mutual_info_score(series, df_full_train.salary)

mi = df_full_train[categorical].apply(mutual_info_salary_score)
mi.sort_values(ascending=False)

# ## Feature importance: Correlation

# corr_w_target = df_full_train[numerical].corrwith(df_full_train.salary.replace({'below_50K':0, 'above_50K':1})).abs().sort_values(ascending=False).to_frame('corr_w_target')
corr_w_target = df_full_train[numerical].corrwith(df_full_train.salary).abs().sort_values(ascending=False).to_frame('corr_w_target')
print(corr_w_target)

# df_full_train2 = df_full_train.copy()
# df_full_train2.salary.replace({'below_50K':0, 'above_50K':1}, inplace=True)

sns.heatmap(df_full_train.corr(numeric_only=True), annot=True)
plt.show()

# ## Feature importance: ROC AUC

result = [(col, roc_auc_score(y_train, df_train[col]) if roc_auc_score(y_train, df_train[col]) >= 0.5 else roc_auc_score(y_train, -df_train[col])) for col in numerical]
result_df = pd.DataFrame(result, columns=['Column', 'AUC'])
print((result_df.sort_values('AUC', ascending=False)))

# ## Apply one-hot-encoding using DictVectorizer

train_dict = df_train[categorical + numerical].to_dict(orient='records')
dv = DictVectorizer(sparse=False)
dv.fit(train_dict)

X_train = dv.transform(train_dict)

val_dict = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dict)

# # Logistic Regression

# ### Logistic Regression Training the model

df_full_train.salary.value_counts(dropna=False)

lr = LogisticRegression(solver='liblinear', max_iter=1000, C=1.0, class_weight='balanced', random_state=42)
lr.fit(X_train, y_train)

y_pred_prob = lr.predict_proba(X_val)[:, 1]
y_pred = lr.predict(X_val) 

score_accuracy = accuracy_score(y_val, y_pred)
score_precision = precision_score(y_val, y_pred, pos_label=1)
score_recall = recall_score(y_val, y_pred, pos_label=1)
score_f1 = f1_score(y_val, y_pred, pos_label=1)
score_roc_auc = roc_auc_score(y_val, y_pred_prob)

print('accuracy_score -->', score_accuracy)
print('precision_score -->', score_precision)
print('recall_score -->', score_recall)
print('f1_score -->', score_f1)
print('roc_auc_score -->', score_roc_auc)

print(confusion_matrix(y_val, y_pred))
ConfusionMatrixDisplay.from_estimator(lr, X_val, y_val, normalize='all')

def eval_metric(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)
    
    print('Val_Set')
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print()
    print('Train_Set')
    print(confusion_matrix(y_train, y_train_pred))
    print(classification_report(y_train, y_train_pred))

eval_metric(lr, X_train, y_train, X_val, y_val)

# ### Logistic Regression Cross Validaiton

def train(df_train, y_train, C=1.0):
    dicts = df_train[columns].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train_ = dv.fit_transform(dicts)

    model = LogisticRegression(solver='liblinear', max_iter=1000, C=C, class_weight='balanced', random_state=42)
    model.fit(X_train_, y_train_)

    return dv, model

def predict(df, dv, model):
    dicts = df[columns].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred

columns = numerical + categorical

scores = []

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train_ = df_train['salary']
    y_val_ = df_val['salary']
    
    del df_train['salary']
    del df_val['salary']

    dv, lr_cv = train(df_train, y_train, C=1.0)
    y_pred = predict(df_val, dv, lr_cv)

    auc = roc_auc_score(y_val_, y_pred)
    scores.append(auc)

print('%.3f +- %.3f' % (np.mean(scores), np.std(scores)))

y_pred = lr_cv.predict_proba(X_train)[:, 1]
auc = roc_auc_score(y_train, y_pred)
print('train_auc:', auc)

y_pred = lr_cv.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_pred)
print('val_auc:', auc)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lr_cv = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, class_weight='balanced', random_state=42)
cv_results = cross_val_score(lr_cv, X_train, y_train, cv = kfold, scoring = 'roc_auc')
round(np.std(cv_results),3)

# ### Logistic Regression Hyperparemeter Tuning

n_splits = 5

for C in tqdm([0.01, 0.1, 0.5, 1, 5, 10]):
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    scores = []

    for train_idx, val_idx in kfold.split(df_full_train):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]

        y_train_ = df_train['salary'].values
        y_val_ = df_val['salary'].values

        dv, lr_gs = train(df_train, y_train_, C=C)
        y_pred = predict(df_val, dv, lr_gs)

        auc = roc_auc_score(y_val_, y_pred)
        scores.append(auc)

    print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))

scores

# model = LogisticRegression(solver='liblinear', max_iter=1000, C=C, class_weight='balanced', random_state=42)
model = LogisticRegression(solver='liblinear', max_iter=1000, class_weight='balanced', random_state=42)
C = [0.01, 0.1, 0.5, 1, 5, 10] 

param_grid = {'C' : C}

cv = StratifiedKFold(n_splits = 5)

lr_gs = GridSearchCV(estimator=model,
                          param_grid=param_grid,
                          cv=cv,
                          scoring = 'roc_auc',
                          n_jobs = -1,
                          return_train_score=True)

lr_gs.fit(X_train, y_train)

print(lr_gs.best_estimator_)
print('-'*25)
print(pd.DataFrame(lr_gs.cv_results_).loc[lr_gs.best_index_, ['mean_test_score', 'mean_train_score']])

# ### Logistic Regression Final Model

log_final = LogisticRegression(solver='liblinear', max_iter=1000, C=0.5, class_weight='balanced', random_state=42)
log_final.fit(X_train, y_train)

y_pred_prob = log_final.predict_proba(X_val)[:, 1]
y_pred = log_final.predict(X_val)

log_accuracy = accuracy_score(y_val, y_pred)
log_precision = precision_score(y_val, y_pred, pos_label=1)
log_recall = recall_score(y_val, y_pred, pos_label=1)
log_f1 = f1_score(y_val, y_pred, pos_label=1)
log_roc_auc = roc_auc_score(y_val, y_pred_prob)

print('accuracy_score -->', log_accuracy)
print('precision_score -->', log_precision)
print('recall_score -->', log_recall)
print('f1_score -->', log_f1)
print('roc_auc_score -->', log_roc_auc)

# ### Logistic Regression Final Model with Feature Scaling

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

log_final_scaled = LogisticRegression(solver='liblinear', max_iter=1000, C=0.5, class_weight='balanced', random_state=42)
log_final_scaled.fit(X_train_scaled, y_train)

y_pred_prob = log_final.predict_proba(X_val_scaled)[:, 1]
y_pred = log_final.predict(X_val_scaled)

score_accuracy = accuracy_score(y_val, y_pred)
score_precision = precision_score(y_val, y_pred, pos_label=1)
score_recall = recall_score(y_val, y_pred, pos_label=1)
score_f1 = f1_score(y_val, y_pred, pos_label=1)
score_roc_auc = roc_auc_score(y_val, y_pred_prob)

print('accuracy_score -->', score_accuracy)
print('precision_score -->', score_precision)
print('recall_score -->', score_recall)
print('f1_score -->', score_f1)
print('roc_auc_score -->', score_roc_auc)

# Scores without scaling is better.

# # Decision Trees

# ### Decision Tree Training the model

dt = DecisionTreeClassifier(class_weight='balanced', random_state=42)
dt.fit(X_train, y_train)

print(export_text(dt, feature_names=list(dv.get_feature_names_out())))

y_pred_prob = dt.predict_proba(X_val)[:, 1]
y_pred = dt.predict(X_val)

score_accuracy = accuracy_score(y_val, y_pred)
score_precision = precision_score(y_val, y_pred, pos_label=1)
score_recall = recall_score(y_val, y_pred, pos_label=1)
score_f1 = f1_score(y_val, y_pred, pos_label=1)
score_roc_auc = roc_auc_score(y_val, y_pred_prob)

print('accuracy_score -->', score_accuracy)
print('precision_score -->', score_precision)
print('recall_score -->', score_recall)
print('f1_score -->', score_f1)
print('roc_auc_score -->', score_roc_auc)

print(confusion_matrix(y_val, y_pred))
ConfusionMatrixDisplay.from_estimator(dt, X_val, y_val, normalize='all')

def eval_metric(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)
    
    print('Val_Set')
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print()
    print('Train_Set')
    print(confusion_matrix(y_train, y_train_pred))
    print(classification_report(y_train, y_train_pred))

eval_metric(dt, X_train, y_train, X_val, y_val)

# ### Decision Tree Cross Validaiton

def train(df_train, y_train, C=1.0):
    dicts = df_train[columns].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train_ = dv.fit_transform(dicts)

    model = DecisionTreeClassifier(class_weight='balanced', random_state=42)
    model.fit(X_train_, y_train_)

    return dv, model

def predict(df, dv, model):
    dicts = df[columns].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred

columns = numerical + categorical

scores = []

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train_ = df_train['salary']
    y_val_ = df_val['salary']
    
    del df_train['salary']
    del df_val['salary']

    dv, dt_cv = train(df_train, y_train, C=1.0)
    y_pred = predict(df_val, dv, dt_cv)

    auc = roc_auc_score(y_val_, y_pred)
    scores.append(auc)

print('%.3f +- %.3f' % (np.mean(scores), np.std(scores)))

y_pred = dt_cv.predict_proba(X_train)[:, 1]
auc = roc_auc_score(y_train, y_pred)
print('train_auc:', auc)

y_pred = dt_cv.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_pred)
print('val_auc:', auc)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
dt_cv = DecisionTreeClassifier(class_weight='balanced', random_state=42)
cv_results = cross_val_score(dt_cv, X_train, y_train, cv = kfold, scoring = 'roc_auc')
round(np.std(cv_results),3)

# ### Decision Tree Hyperparemeter Tuning

scores = []

for depth in tqdm([1, 2, 4, 5, 6, 7, 10, 15, 20]):
    for s in tqdm([1, 2, 5, 10, 15, 20, 50, 100, 200, 500]):
        dt = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=s, class_weight='balanced', random_state=42)
        dt.fit(X_train, y_train)

        y_pred = dt.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)
        
        scores.append((depth, s, auc))

columns = ['max_depth', 'min_samples_leaf', 'auc']
df_scores = pd.DataFrame(scores, columns=columns)

for d in [1, 2, 4, 5, 6, 7, 10, 15, 20]:
    df_subset = df_scores[df_scores.max_depth == d]
    
    plt.plot(df_subset.min_samples_leaf, df_subset.auc,
             label='max_depth=%d' % d)
    plt.xlabel('min_samples_leaf')
    plt.ylabel('auc')
    plt.legend();

df_scores_pivot = df_scores.pivot(index='min_samples_leaf', columns=['max_depth'], values=['auc'])
df_scores_pivot.round(3)

sns.heatmap(df_scores_pivot, annot=True, fmt='.3f');

# model = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=s, class_weight='balanced', random_state=42)
model = DecisionTreeClassifier(class_weight='balanced', random_state=42)
max_depth = [1, 2, 4, 5, 6, 7, 10, 15, 20]
min_samples_leaf = [1, 2, 5, 10, 15, 20, 50, 100, 200, 500]

param_grid = {'max_depth' : max_depth,
              'min_samples_leaf' : min_samples_leaf}

cv = StratifiedKFold(n_splits = 5)

dt_gs = GridSearchCV(estimator=model,
                     param_grid=param_grid,
                     cv=cv,
                     scoring = 'roc_auc',
                     n_jobs = -1,
                     return_train_score=True)

dt_gs.fit(X_train, y_train)

print(dt_gs.best_estimator_)
print('-'*25)
print(pd.DataFrame(dt_gs.cv_results_).loc[dt_gs.best_index_, ['mean_test_score', 'mean_train_score']])

# ### Decision Tree Final Model

dt_final = DecisionTreeClassifier(max_depth=20, min_samples_leaf=100, class_weight='balanced', random_state=42)
dt_final.fit(X_train, y_train)

y_pred_prob = dt_final.predict_proba(X_val)[:, 1]
y_pred = dt_final.predict(X_val)

dt_accuracy = accuracy_score(y_val, y_pred)
dt_precision = precision_score(y_val, y_pred, pos_label=1)
dt_recall = recall_score(y_val, y_pred, pos_label=1)
dt_f1 = f1_score(y_val, y_pred, pos_label=1)
dt_roc_auc = roc_auc_score(y_val, y_pred_prob)

print('accuracy_score -->', dt_accuracy)
print('precision_score -->', dt_precision)
print('recall_score -->', dt_recall)
print('f1_score -->', dt_f1)
print('roc_auc_score -->', dt_roc_auc)


# ### Decision Tree Feature Importance

features = dv.get_feature_names_out()
dt_imp = pd.DataFrame(data = dt_final.feature_importances_, 
                    index=dv.get_feature_names_out(), 
                    columns = ["Feature Importance"]).sort_values("Feature Importance", ascending=False)
dt_imp[:21]

ax = dt_imp[:20].plot(kind='bar', figsize=(20,10))
ax.bar_label(ax.containers[0],fmt="%.3f")
plt.xticks(rotation = 75);

# # Random Forest

# ### Random Forest Training the model

rf = RandomForestClassifier(class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)

print(export_text(dt, feature_names=list(dv.get_feature_names_out())))

y_pred_prob = rf.predict_proba(X_val)[:, 1]
y_pred = rf.predict(X_val)

score_accuracy = accuracy_score(y_val, y_pred)
score_precision = precision_score(y_val, y_pred, pos_label=1)
score_recall = recall_score(y_val, y_pred, pos_label=1)
score_f1 = f1_score(y_val, y_pred, pos_label=1)
score_roc_auc = roc_auc_score(y_val, y_pred_prob)

print('accuracy_score -->', score_accuracy)
print('precision_score -->', score_precision)
print('recall_score -->', score_recall)
print('f1_score -->', score_f1)
print('roc_auc_score -->', score_roc_auc)

print(confusion_matrix(y_val, y_pred))
ConfusionMatrixDisplay.from_estimator(rf, X_val, y_val, normalize='all')

def eval_metric(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)
    
    print('Val_Set')
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print()
    print('Train_Set')
    print(confusion_matrix(y_train, y_train_pred))
    print(classification_report(y_train, y_train_pred))

eval_metric(rf, X_train, y_train, X_val, y_val)

# ### Random Forest Cross Validaiton

def train(df_train, y_train, C=1.0):
    dicts = df_train[columns].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train_ = dv.fit_transform(dicts)

    model = RandomForestClassifier(class_weight='balanced', random_state=42)
    model.fit(X_train_, y_train_)

    return dv, model

def predict(df, dv, model):
    dicts = df[columns].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred

columns = numerical + categorical

scores = []

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train_ = df_train['salary']
    y_val_ = df_val['salary']
    
    del df_train['salary']
    del df_val['salary']

    dv, rf_cv = train(df_train, y_train, C=1.0)
    y_pred = predict(df_val, dv, rf_cv)

    auc = roc_auc_score(y_val_, y_pred)
    scores.append(auc)

print('%.3f +- %.3f' % (np.mean(scores), np.std(scores)))

y_pred = rf_cv.predict_proba(X_train)[:, 1]
auc = roc_auc_score(y_train, y_pred)
print('train_auc:', auc)

y_pred = rf_cv.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_pred)
print('val_auc:', auc)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf_cv = RandomForestClassifier(class_weight='balanced', random_state=42)
cv_results = cross_val_score(rf_cv, X_train, y_train, cv = kfold, scoring = 'roc_auc')
round(np.std(cv_results),3)

# ### Random Forest Hyperparemeter Tuning

scores = []

for d in [10, 15, 20, 25]:
    for n in range(10, 201, 10):
        rf = RandomForestClassifier(n_estimators=n,
                                    max_depth=d,
                                    random_state=42,
                                    class_weight='balanced')
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_val)
        auc = roc_auc_score(y_val, y_pred)

        scores.append((d, n, auc))

columns = ['max_depth', 'n_estimators', 'auc']
df_scores = pd.DataFrame(scores, columns=columns)

for d in [10, 15, 20, 25]:
    df_subset = df_scores[df_scores.max_depth == d]
    
    plt.plot(df_subset.n_estimators, df_subset.auc,
             label='max_depth=%d' % d)
    plt.xlabel('Num of Tress')
    plt.ylabel('auc')
    plt.legend();

df_scores_pivot = df_scores.pivot(index='n_estimators', columns=['max_depth'], values=['auc'])
df_scores_pivot.round(3)

sns.heatmap(df_scores_pivot, annot=True, fmt='.3f');

# model = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, class_weight='balanced', random_state=42)
model = RandomForestClassifier(class_weight='balanced', random_state=42)
max_depth = [1, 2, 4, 5, 6, 7, 10, 15, 20]
n_estimators = [n for n in range(10, 201, 10)]

param_grid = {'max_depth' : max_depth,
              'n_estimators' : n_estimators}

cv = StratifiedKFold(n_splits = 5)

rf_gs = GridSearchCV(estimator=model,
                     param_grid=param_grid,
                     cv=cv,
                     scoring = 'roc_auc',
                     n_jobs = -1,
                     return_train_score=True)

rf_gs.fit(X_train, y_train)

print(rf_gs.best_estimator_)
print('-'*25)
print(pd.DataFrame(rf_gs.cv_results_).loc[rf_gs.best_index_, ['mean_test_score', 'mean_train_score']])

# ### Random Forest Final Model

rf_final = RandomForestClassifier(class_weight='balanced', max_depth=15, n_estimators=130, random_state=42)
rf_final.fit(X_train, y_train)

y_pred_prob = rf_final.predict_proba(X_val)[:, 1]
y_pred = rf_final.predict(X_val)

rf_accuracy = accuracy_score(y_val, y_pred)
rf_precision = precision_score(y_val, y_pred, pos_label=1)
rf_recall = recall_score(y_val, y_pred, pos_label=1)
rf_f1 = f1_score(y_val, y_pred, pos_label=1)
rf_roc_auc = roc_auc_score(y_val, y_pred_prob)

print('accuracy_score -->', rf_accuracy)
print('precision_score -->', rf_precision)
print('recall_score -->', rf_recall)
print('f1_score -->', rf_f1)
print('roc_auc_score -->', rf_roc_auc)

# ### Random Forest Feature Importance

features = dv.get_feature_names_out()
rf_imp = pd.DataFrame(data = rf_final.feature_importances_, 
                      index=dv.get_feature_names_out(), 
                      columns = ["Feature Importance"]).sort_values("Feature Importance", ascending=False)
rf_imp

ax = rf_imp.plot(kind='bar', figsize=(20,10))
ax.bar_label(ax.containers[0],fmt="%.3f")
plt.xticks(rotation = 75);

# # XGBoost

# ### XGBoost Training the model

features = dv.get_feature_names_out()
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)

xgb_params = {'eta': 0.3, 
              'max_depth': 6,
              'min_child_weight': 1,
              'objective': 'binary:logistic',
              'nthread': 8,
              'seed': 42,
              'verbosity': 1,
              'scale_pos_weight' : 3
             }

xgboost = xgb.train(xgb_params, dtrain, num_boost_round=100)

y_pred = xgboost.predict(dval)

actual_positive = (y_val == 1)
actual_negative = (y_val == 0)

threshold = 0.5
predict_positive = (y_pred >= threshold)
predict_negative = (y_pred < threshold)

tp = (predict_positive & actual_positive).sum()
tn = (predict_negative & actual_negative).sum()

fp = (predict_positive & actual_negative).sum()
fn = (predict_negative & actual_positive).sum()

score_accuracy = (tp+tn)/(tp+tn+fp+fn)
score_precision = tp / (tp + fp)
score_recall = tp / (tp + fn)
score_f1 = 2 * (score_precision * score_recall) / (score_precision + score_recall)
score_roc_auc = roc_auc_score(y_val, y_pred)

print('accuracy_score -->', score_accuracy)
print('precision_score -->', score_precision)
print('recall_score -->', score_recall)
print('f1_score -->', score_f1)
print('roc_auc_score -->', score_roc_auc)

confusion_matrix = np.array([
                             [tn, fp],
                             [fn, tp]
                            ])
print(confusion_matrix)
print('---'*5)
print((confusion_matrix / confusion_matrix.sum()).round(2))

# ### XGBoost Hyperparemeter Tuning

y_pred = xgboost.predict(dval)
roc_auc_score(y_val, y_pred)

watchlist = [(dtrain, 'train'), (dval, 'val')]

get_ipython().run_cell_magic('capture', 'output', "\nxgb_params = {'eta': 0.3, \n              'max_depth': 6,\n              'min_child_weight': 1,\n              \n              'objective': 'binary:logistic',\n              'eval_metric': 'auc',\n              \n              'nthread': 8,\n              'seed': 42,\n              'verbosity': 1,\n              'scale_pos_weight' : 3\n             }\n\nxgboost = xgb.train(xgb_params, \n                    dtrain, \n                    num_boost_round=200,\n                    verbose_eval=5,\n                    evals=watchlist)\n")

s = output.stdout

print(s[:200])

def parse_xgb_output(output):
    results = []

    for line in output.stdout.strip().split('\n'):
        it_line, train_line, val_line = line.split('\t')

        it = int(it_line.strip('[]'))
        train = float(train_line.split(':')[1])
        val = float(val_line.split(':')[1])

        results.append((it, train, val))
    
    columns = ['num_iter', 'train_auc', 'val_auc']
    df_results = pd.DataFrame(results, columns=columns)
    return df_results

df_score = parse_xgb_output(output)
df_score

plt.plot(df_score.num_iter, df_score.train_auc, label='train')
plt.plot(df_score.num_iter, df_score.val_auc, label='val')
plt.legend();

plt.plot(df_score.num_iter, df_score.val_auc, label='val')
plt.legend();

# **eta**

scores = {} #Try for different eta values --> 'eta=0.01', 'eta=0.05', 'eta=0.1', 'eta=0.3', 'eta=1.0'

get_ipython().run_cell_magic('capture', 'output', "\nxgb_params = {\n                'eta': 0.3, \n                'max_depth': 6,\n                'min_child_weight': 1,\n\n                'objective': 'binary:logistic',\n                'eval_metric': 'auc',\n\n                'nthread': 8,\n                'seed': 42,\n                'verbosity': 1,\n                'scale_pos_weight': 3\n            }\n\nxgboost = xgb.train(xgb_params, \n                    dtrain, \n                    num_boost_round=200,\n                    verbose_eval=5,\n                    evals=watchlist)\n")

'eta=%s' % (xgb_params['eta'])

key = 'eta=%s' % (xgb_params['eta'])
scores[key] = parse_xgb_output(output)
key

scores
# scores['eta=0.01']

scores.keys()

for key, df_score in scores.items() :
    plt.plot(df_score.num_iter, df_score.val_auc, label=key)
    plt.legend();

etas = ['eta=0.05', 'eta=0.1']

for eta in etas :
    df_score = scores[eta]
    plt.plot(df_score.num_iter, df_score.val_auc, label=eta)
    plt.legend()

# Best eta value --> 0.05

# **max_depth**

scores = {} # Try for different max_depth values --> 'max_depth=3', 'max_depth=4', 'max_depth=6', 'max_depth=10', 'max_depth=20'

get_ipython().run_cell_magic('capture', 'output', "\nxgb_params = {\n                'eta': 0.05, \n                'max_depth': 20,\n                'min_child_weight': 1,\n\n                'objective': 'binary:logistic',\n                'eval_metric': 'auc',\n\n                'nthread': 8,\n                'seed': 42,\n                'verbosity': 1,\n                'scale_pos_weight': 3\n            }\n\nxgboost = xgb.train(xgb_params, \n                    dtrain, \n                    num_boost_round=200,\n                    verbose_eval=5,\n                    evals=watchlist)\n")

'max_depth=%s' % (xgb_params['max_depth'])

key = 'max_depth=%s' % (xgb_params['max_depth'])
scores[key] = parse_xgb_output(output)
key

scores
# scores['max_depth=3']

scores.keys()

for key, df_score in scores.items() :
    plt.plot(df_score.num_iter, df_score.val_auc, label=key)
    plt.legend();

max_depths = ['max_depth=4', 'max_depth=6', 'max_depth=10']

for max_depth in max_depths :
    df_score = scores[max_depth]
    plt.plot(df_score.num_iter, df_score.val_auc, label=max_depth)
    plt.legend()

max_depths = ['max_depth=4', 'max_depth=6', 'max_depth=10']

for max_depth in max_depths :
    df_score = scores[max_depth]
    plt.plot(df_score.num_iter, df_score.val_auc, label=max_depth)
    plt.ylim(0.825, 0.925)
    plt.legend()

# Best max_depth value --> 6

# **min_child_weight**

scores = {} # Try for different min_child_weight values --> 'min_child_weight=1, 5, 10, 20, 30'

get_ipython().run_cell_magic('capture', 'output', "\nxgb_params = {\n                'eta': 0.05, \n                'max_depth': 6,\n                'min_child_weight': 30,\n\n                'objective': 'binary:logistic',\n                'eval_metric': 'auc',\n\n                'nthread': 8,\n                'seed': 42,\n                'verbosity': 1,\n                'scale_pos_weight': 3\n            }\n\nxgboost = xgb.train(xgb_params, \n                    dtrain, \n                    num_boost_round=200,\n                    verbose_eval=5,\n                    evals=watchlist)\n")

'min_child_weight=%s' % (xgb_params['min_child_weight'])

key = 'min_child_weight=%s' % (xgb_params['min_child_weight'])
scores[key] = parse_xgb_output(output)
key

scores
# scores['min_child_weight=1']

scores.keys()

for key, df_score in scores.items() :
    plt.plot(df_score.num_iter, df_score.val_auc, label=key)
    plt.legend();

min_child_weights = ['min_child_weight=1', 'min_child_weight=5', 'min_child_weight=10']

for min_child_weight in min_child_weights :
    df_score = scores[min_child_weight]
    plt.plot(df_score.num_iter, df_score.val_auc, label=min_child_weight)
    plt.legend()

# Best min_child_weight value --> 5

# ### XGBoost Final Model

xgb_params = { 'eta': 0.05, 
               'max_depth': 6,
               'min_child_weight': 5,
    
               'objective': 'binary:logistic',
               'eval_metric': 'auc',

               'nthread': 8,
               'seed': 42,
               'verbosity': 1,
               'scale_pos_weight': 3
             }

xgb_final = xgb.train(xgb_params, dtrain, num_boost_round=200)

y_pred = xgb_final.predict(dval)

actual_positive = (y_val == 1)
actual_negative = (y_val == 0)

threshold = 0.5
predict_positive = (y_pred >= threshold)
predict_negative = (y_pred < threshold)

tp = (predict_positive & actual_positive).sum()
tn = (predict_negative & actual_negative).sum()

fp = (predict_positive & actual_negative).sum()
fn = (predict_negative & actual_positive).sum()

xgb_accuracy = (tp+tn)/(tp+tn+fp+fn)
xgb_precision = tp / (tp + fp)
xgb_recall = tp / (tp + fn)
xgb_f1 = 2 * (score_precision * score_recall) / (score_precision + score_recall)
xgb_roc_auc = roc_auc_score(y_val, y_pred)

print('accuracy_score -->', xgb_accuracy)
print('precision_score -->', xgb_precision)
print('recall_score -->', xgb_recall)
print('f1_score -->', xgb_f1)
print('roc_auc_score -->', xgb_roc_auc)

confusion_matrix = np.array([
                             [tn, fp],
                             [fn, tp]
                            ])
print(confusion_matrix)
print('---'*5)
print((confusion_matrix / confusion_matrix.sum()).round(2))

# ### XGBoost Feature Importance

feature_important = xgb_final.get_score(importance_type='weight')
keys = list(feature_important.keys())
values = list(feature_important.values())

xgb_imp = pd.DataFrame(data=values, index=keys, columns=["score"])
xgb_imp.nlargest(43, columns="score").plot(kind='barh', figsize = (20,20));

# # Final Model

# ### Comparison of models

compare = pd.DataFrame({'Model': ['LogReg_model', 'DT_model', 'RF_model', 'XGB_model'],
                        'Accuracy_Score': [log_accuracy, dt_accuracy, rf_accuracy, xgb_accuracy],
                        'Precision_Score': [log_precision, dt_precision, rf_precision, xgb_precision],
                        'F1_Score': [log_f1, dt_f1, rf_f1, xgb_f1],
                        'Recall_Score': [log_recall, dt_recall, rf_recall, xgb_recall],
                        'ROC_AUC_Score': [log_roc_auc, dt_roc_auc, rf_roc_auc, xgb_roc_auc]})

compare = compare.sort_values(by='Accuracy_Score', ascending=True)
fig = px.bar(compare, x = 'Accuracy_Score', y = 'Model', title = 'Accuracy_Score')
fig.show()

compare = compare.sort_values(by='Precision_Score', ascending=True)
fig = px.bar(compare, x = 'Precision_Score', y = 'Model', title = 'Precision_Score')
fig.show()

compare = compare.sort_values(by='F1_Score', ascending=True)
fig = px.bar(compare, x = 'F1_Score', y = 'Model', title = 'F1_Score')
fig.show()

compare = compare.sort_values(by='Recall_Score', ascending=True)
fig = px.bar(compare, x = 'Recall_Score', y = 'Model', title = 'Recall_Score')
fig.show()

compare = compare.sort_values(by='ROC_AUC_Score', ascending=True)
fig = px.bar(compare, x = 'ROC_AUC_Score', y = 'Model', title = 'ROC_AUC_Score')
fig.show()

compare.T

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

output_file = 'model_rf'
output_file

f_out = open(output_file, 'wb')
pickle.dump((dv, model), f_out)
f_out.close()

model_file = 'model_rf'

with open(model_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

# ### Load the model

model_file = 'model_rf'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

dv, model

# ### Prediction

df.sample(1).to_dict(orient='records')

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
