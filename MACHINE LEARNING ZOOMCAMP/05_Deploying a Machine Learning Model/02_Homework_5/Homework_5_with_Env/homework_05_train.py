#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.feature_extraction import DictVectorizer

from sklearn.linear_model import LogisticRegression

# !pip install tqdm
from tqdm.auto import tqdm

import pickle

# Parameters
C = 0.1
n_splits = 5
output_file = f'model_C={C}.bin'


# Data preparation

df0 = pd.read_csv("bank.csv", sep=';')
df = df0.copy()
df.head()

df.shape

df.columns

df.info()

df.describe()

df.describe(include = object)

df.isnull().sum()

df.duplicated().sum()

numerical = df.drop(['y'], axis=1).select_dtypes('number').columns.to_list()

categorical = df.drop(['y'], axis=1).select_dtypes('object').columns.to_list()

print(numerical)
print()
print(categorical)


# **Make the "y" binary**

df.y = (df.y =="yes").astype(int)
df.head()

df.y.value_counts(dropna=False)


# **Split the data**

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

len(df_full_train), len(df_test)

df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

len(df_train), len(df_val), len(df_test)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

df_train.head()

df_val.head()

df_test.head()

y_train_ = df_train.y.values
y_val_ = df_val.y.values
y_test_ = df_test.y.values

len(y_train_), len(y_val_), len(y_test_)

print(y_train_[:25])
print(y_val_[:25])
print(y_test_[:25])

del df_train["y"]
del df_val["y"]
del df_test["y"]


# ### ROC AUC feature importance

result = [(col, roc_auc_score(y_train_, df_train[col]) if roc_auc_score(y_train_, df_train[col]) >= 0.5 else roc_auc_score(y_train_, -df_train[col])) for col in numerical]
result_df = pd.DataFrame(result, columns=['Column', 'AUC'])
print((result_df.sort_values("AUC", ascending=False)))


# ### Training the model

# **Apply one-hot-encoding using DictVectorizer**

train_dict = df_train[categorical + numerical].to_dict(orient='records')
train_dict[0]

dv = DictVectorizer(sparse=False)
dv.fit(train_dict)

X_train_ = dv.transform(train_dict)
X_train_

X_train_.shape

dv.get_feature_names_out()

val_dict = df_val[categorical + numerical].to_dict(orient='records')
X_val_ = dv.transform(val_dict)
X_val_


# **Train the Logistic Regression**

model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
model.fit(X_train_, y_train_)

model.intercept_[0]

model.coef_[0].round(3)

pd.options.display.float_format = '{:.3f}'.format

y_pred = model.predict_proba(X_val_)[:, 1]
y_pred

score_roc_auc = round(roc_auc_score(y_val_, y_pred), 3)
print("roc_auc_score -->", score_roc_auc)


# ### Precision and Recall

# **Compute precision and recall**

thresholds = np.arange(0.0, 1.0, 0.01)

score_pre_rec = []

for t in thresholds:
    score_precision = precision_score(y_val_, y_pred >= t)
    score_recall = recall_score(y_val_, y_pred >= t)
    score_pre_rec.append({score_precision, score_recall})

scores = pd.DataFrame(score_pre_rec, columns = ["Precision", "Recall"])
scores["Threshold"] = thresholds

last_column = scores.pop('Threshold')
scores.insert(0, 'Threshold', last_column)

scores.sort_values("Threshold")

plt.figure(figsize=(10,6))

plt.plot(scores.Threshold, scores.Precision)
plt.plot(scores.Threshold, scores.Recall)
plt.legend(["Precision", "Recall"])
plt.xlabel("Threshold Value")
plt.ylabel("Precision & Recall Scores")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.show()


# ### F1 score

scores["f1_manual"] = 2 * scores.Precision * scores.Recall / (scores.Precision + scores.Recall)
scores

thresholds = np.arange(0.0, 1.0, 0.01)

f1_sklearn = []

for t in thresholds:
    score_f1 = f1_score(y_val_, y_pred >= t)
    f1_sklearn.append(score_f1)
    
scores["f1_sklearn"] = f1_sklearn
scores

plt.figure(figsize=(10,6))

plt.plot(scores.Threshold, scores.f1_manual)
plt.xlabel("Threshold Value")
plt.ylabel("F1 Scores")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.show()

scores.loc[scores['f1_manual'].idxmax()]

scores.sort_values("f1_manual", ascending=False).head(1)

scores.sort_values("f1_sklearn", ascending=False).head(1)


# ### 5-Fold CV

def train(df_train, y_train, C=1.0):
    dicts = df_train[columns].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(solver='liblinear', max_iter=1000, C=C)
    model.fit(X_train, y_train)

    return dv, model

def predict(df, dv, model):
    dicts = df[columns].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred

columns = numerical + categorical

scores = []

kfold = KFold(n_splits=5, shuffle=True, random_state=1)

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train["y"]
    y_val = df_val["y"]
    
    del df_train["y"]
    del df_val["y"]

    dv, model = train(df_train, y_train, C=1.0)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

print('%.3f +- %.3f' % (np.mean(scores), np.std(scores)))


kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
model_cv = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
cv_results = cross_val_score(model_cv, X_train_, y_train_, cv = kfold, scoring = "roc_auc")
round(np.std(cv_results),3)


# ### Hyperparemeter Tuning

n_splits = 5

for C in tqdm([0.01, 0.1, 0.5, 10]):
    kfold = KFold(n_splits=5, shuffle=True, random_state=1)

    scores = []

    for train_idx, val_idx in kfold.split(df_full_train):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]

        y_train = df_train["y"].values
        y_val = df_val["y"].values

        dv, model = train(df_train, y_train, C=C)
        y_pred = predict(df_val, dv, model)

        auc = roc_auc_score(y_val, y_pred)
        scores.append(auc)

    print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))


model = LogisticRegression(solver='liblinear', max_iter=1000, C=C)

C = [0.01, 0.1, 0.5, 10] 

param_grid = {"C" : C}

cv = StratifiedKFold(n_splits = 5)

grid_model = GridSearchCV(estimator=model,
                          param_grid=param_grid,
                          cv=cv,
                          scoring = "roc_auc",
                          n_jobs = -1,
                          return_train_score=True)

grid_model.fit(X_train_, y_train_)

print(grid_model.best_estimator_)
print("-"*25)
print(pd.DataFrame(grid_model.cv_results_).loc[grid_model.best_index_, ["mean_test_score", "mean_train_score"]])

# Training the final model

C=0.1

dv, model = train(df_full_train, df_full_train.y.values, C=C)
y_pred = predict(df_test, dv, model)

# y_test = df_test["y"].values
auc = roc_auc_score(y_test_, y_pred)
auc


# **Save the model**

output_file

f_out = open(output_file, 'wb')
pickle.dump((dv, model), f_out)
f_out.close()

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)


# **Load the model**

with open(output_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

dv, model

customer = {'age' : 41, 
            'job' : 'management', 
            'marital' : 'single', 
            'education' : 'unknown', 
            'default' : 'no', 
            'balance' : 1422, 
            'housing' : 'yes',
            'loan' : 'no', 
            'contact' : 'unknown', 
            'day' : 22, 
            'month' : 'nov', 
            'duration' : 278, 
            'campaign' : 4, 
            'pdays' : 43,
            'previous' : 3, 
            'poutcome' : 'failure'}


X = dv.transform([customer])

model.predict_proba(X)[0,1]