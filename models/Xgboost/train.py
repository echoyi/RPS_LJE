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


import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

import config

warnings.filterwarnings("ignore")


def load_data():
    root_path = config.project_root
    path = '{}/models/Xgboost/data'.format(root_path)
    X_train = pd.read_csv('{}/X_train_clean_res.csv'.format(path), index_col=0)
    y_train = pd.read_csv('{}/Y_train_clean_res.csv'.format(path), index_col=0)
    X_test = pd.read_csv('{}/X_test_clean.csv'.format(path), index_col=0)
    y_test = pd.read_csv('{}/Y_test_clean.csv'.format(path), index_col=0)
    X_train = X_train.drop('id', axis=1)
    X_test = X_test.drop('id', axis=1)
    return X_train, y_train, X_test, y_test


def calculate_intermediate_values(model, data):
    intermediate_x = []
    for i in range(0, model.n_estimators):
        p = model.predict_proba(data, iteration_range=[i, i + 1])[:, 1]
        intermediate_x.append(p)
    intermediate_x = np.stack(intermediate_x, axis=1)
    intermediate_x = np.concatenate([intermediate_x,
                                     np.ones((len(intermediate_x), 1))], axis=1)
    return intermediate_x


def train(X_train, y_train, X_test, y_test, path, save=False, seed=None):
    params = {
        'n_estimators': 80,
        'objective': 'binary:logistic',
        'learning_rate': 0.005,
        'gamma': 0.01,
        'colsample_bytree': 0.5,
        'min_child_weight': 3,
        'max_depth': 8,
        'eval_metric': 'logloss',
        'seed': 9999,
        'n_jobs': -1
    }
    if seed is not None:
        params['seed'] = seed
    #
    if os.path.exists('{}/model/xgb.pt'.format(path)):
        model = pickle.load(open('{}/model/xgb.pt'.format(path), 'rb'))
    else:
        model = XGBClassifier(**params).fit(X_train, y_train.values.ravel(),
                                            eval_metric='auc', eval_set=[(X_train, y_train.values.ravel())],
                                            early_stopping_rounds=100, verbose=True)
        model.set_params(**{'n_estimators': model.best_ntree_limit})
        model.fit(X_train, y_train.values.ravel(), verbose=False)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    test_accuracy = accuracy_score(y_test.values.ravel(), pred_test)
    train_accuracy = accuracy_score(y_train.values.ravel(), pred_train)
    print("Training accuracy:{}, Testing accuracy:{}, number of model estimaters: {}".format(train_accuracy,
                                                                                             test_accuracy,
                                                                                             model.n_estimators))

    if save:
        intermediate_train = calculate_intermediate_values(model, X_train)
        intermediate_test = calculate_intermediate_values(model, X_test)
        pickle.dump(model, open('{}/model/xgb.pt'.format(path), 'wb'))
        w = np.concatenate([np.ones(model.n_estimators), np.array(-0.5 * (model.n_estimators - 1)).reshape(1)]).reshape(
            1, -1)
        np.savez('{}/model/saved_outputs'.format(path),
                 intermediate_train=intermediate_train, pred_train=pred_train,
                 labels_train=y_train.values.ravel(), outputs_train=model.predict_proba(X_train),
                 intermediate_test=intermediate_test, pred_test=pred_test,
                 labels_test=y_test.values.ravel(), outputs_test=model.predict_proba(X_test), weight=w,
                 train_accuracy=train_accuracy, test_accuracy=test_accuracy)
    return test_accuracy, model


def train_with_flipped(flipped_idx, path, save=False, portion=0.2, num_of_samples=1118):
    if flipped_idx is None:
        flipped_idx = np.load('{}/orders/random_flipping_order.npy'.format(path))[:int(portion * num_of_samples)]
    X_train, y_train, X_test, y_test = load_data()
    if len(flipped_idx) > 0:
        y_train.values[flipped_idx] = 1 - y_train.values[flipped_idx]
    test_accuracy, model = train(X_train, y_train, X_test, y_test, path, save)
    return test_accuracy


def train_and_record_accuracies(path, method, save=False, portion=0.2,
                                num_of_samples=1118, average_over_runs=3, experiment_range=8):
    test_accuracies = []
    inputs = np.load('{}/orders/order_{}.npz'.format(path, method), allow_pickle=True)['inputs']
    for idx in inputs[:experiment_range]:
        test_accuracy = 0
        for i in range(average_over_runs):
            acc = train_with_flipped(idx, path, save=save, portion=portion, num_of_samples=num_of_samples)
            test_accuracy += acc
        test_accuracies.append(test_accuracy / average_over_runs)
    test_accuracies = np.stack(test_accuracies)
    np.save('{}/accuracies/accuracies_{}'.format(path, method), test_accuracies)


def run():
    X_train, y_train, X_test, y_test = load_data()
    train(X_train, y_train, X_test, y_test, 'saved_models/base', save=True, seed=9999)


if __name__ == "__main__":
    run()
