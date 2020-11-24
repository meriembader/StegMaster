import os

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from model import model
import joblib


a = 0
def lgb_accuracy_score(y_true, y_hat):
    global a
    best_score = 0
    best_thresh = 0
    for thresh in np.linspace(0, 0.3, 300):
        yh = y_hat > thresh
        score = accuracy_score(y_true, yh)
        if score > best_score:
            best_score = score
            best_thresh = thresh
        a=best_thresh
    return 'acc_',  accuracy_score(y_true, y_hat > best_thresh), True


data_dir = "../data/"

if __name__ == "__main__":
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    validation = pd.read_csv(os.path.join(data_dir, "validation.csv"))
    cols = [col for col in train.columns if col not in [
        "target", "client_id", "creation_date"]]

    model.fit(
        train[cols].values, train.target.values,
        eval_set=[(validation[cols].values, validation.target.values)],
        eval_metric=lgb_accuracy_score,
        verbose=50,
        early_stopping_rounds=100)
    print(a)
    # model.save_model('../model/lgb_classifier.txt', num_iteration=model.best_iteration) 
    joblib.dump(model, '../model/final_model.pkl')
