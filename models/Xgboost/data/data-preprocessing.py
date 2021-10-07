'''
modified from https://www.kaggle.com/hendraherviawan/predicting-german-credit-default
Copyright [2018] [M HENDRA HERVIAWAN]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

from collections import defaultdict

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import config

root_path = config.project_root
path = '{}/models/Xgboost/data'.format(root_path)
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

names = ['existingchecking', 'duration', 'credithistory', 'purpose', 'creditamount',
         'savings', 'employmentsince', 'installmentrate', 'statussex', 'otherdebtors',
         'residencesince', 'property', 'age', 'otherinstallmentplans', 'housing',
         'existingcredits', 'job', 'peopleliable', 'telephone', 'foreignworker', 'classification']

data = pd.read_csv(url, names=names, delimiter=' ')
data.to_csv('{}/german_data.csv'.format(path))
numvars = ['creditamount', 'duration', 'installmentrate', 'residencesince', 'age',
           'existingcredits', 'peopleliable', 'classification']
catvars = ['existingchecking', 'credithistory', 'purpose', 'savings', 'employmentsince',
           'statussex', 'otherdebtors', 'property', 'otherinstallmentplans', 'housing', 'job',
           'telephone', 'foreignworker']

d = defaultdict(LabelEncoder)

# Encoding the variable
lecatdata = data[catvars].apply(lambda x: d[x.name].fit_transform(x))
# One hot encoding, create dummy variables for every category of every categorical variable
dummyvars = pd.get_dummies(data[catvars])  # %%
data_clean = pd.concat([data[numvars], dummyvars], axis=1)
X_clean = data_clean.drop('classification', axis=1)
y_clean = data_clean['classification']
X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(X_clean, y_clean, test_size=0.2,
                                                                            random_state=1)
X_train_clean = X_train_clean.rename_axis('id').reset_index()
X_test_clean = X_test_clean.rename_axis('id').reset_index()

sm = SMOTE(sampling_strategy='auto')
X_train_clean_res, y_train_clean_res = sm.fit_resample(X_train_clean, y_train_clean)
X_train_clean_res = pd.DataFrame(X_train_clean_res, columns=X_train_clean.keys())

X_train_clean_res.to_csv('{}/X_train_clean_res.csv'.format(path))
y_train_clean_res.to_csv('{}/Y_train_clean_res.csv'.format(path))
X_test_clean.to_csv('{}/X_test_clean.csv'.format(path))
y_test_clean.to_csv('{}/Y_test_clean.csv'.format(path))
X_train_clean.to_csv('{}/X_train_clean.csv'.format(path))
y_train_clean.to_csv('{}/Y_train_clean.csv'.format(path))
