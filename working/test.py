import datetime
import os
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

import lightgbm as lgb
import joblib

from flask import Flask, request, jsonify
import os
from flask_cors import CORS


app = Flask(__name__)

CORS(app)


warnings.filterwarnings("ignore")

def predict(file1, file2):
    model = joblib.load('../model/final_model.pkl')

    test = file_to_pandas(file1)
    invoice = file_to_pandas(file2)

    l = ['tarif_type', 'counter_number',
       'counter_statue', 'counter_code', 'reading_remarque',
       'counter_coefficient', 'consommation_level_1', 'consommation_level_2',
       'consommation_level_3', 'consommation_level_4', 'old_index',
       'new_index', 'months_number']
    for col in l:
        invoice[col] = invoice[col].astype(int)
    print("holmaaaaaaaaaaaaaaaaaaa")

    l = ['disrict', 'client_catg', 'region']

    for col in l:
        test[col] = test[col].astype(int)

    print("holmaaaaaaaaaaaaaaaaaaa")




    invoice["diff_index"] = invoice["new_index"]-invoice["old_index"]
    invoice["ration_diff_index"] = (
        invoice["new_index"]-invoice["old_index"])/invoice["months_number"].apply(lambda x: max(x, 36))


    test.creation_date = pd.to_datetime(test.creation_date)
    test['year'] = test['creation_date'].dt.year
    test['month'] = test['creation_date'].dt.month
    test['dow'] = test['creation_date'].dt.dayofweek
    test['day'] = test['creation_date'].dt.day
    test['doy'] = test['creation_date'].dt.dayofyear
    test['woy'] = test['creation_date'].dt.weekofyear
    test['mend'] = test['creation_date'].dt.is_month_end
    test['mstart'] = test['creation_date'].dt.is_month_start
    test['quarter'] = test['creation_date'].dt.quarter
    test['quartstart'] = test['creation_date'].dt.is_quarter_start
    test['quartend'] = test['creation_date'].dt.is_quarter_end

    cols = ['tarif_type', 'counter_number', 'counter_code', 'reading_remarque',
            'counter_coefficient', 'consommation_level_1', 'consommation_level_2',
            'consommation_level_3', 'consommation_level_4', 'old_index',
            'new_index', 'months_number', 'diff_index',
            'ration_diff_index']

    # mean
    print("Mean calculating")
    start = time.time()
    group_test = invoice.groupby("client_id")[cols].mean()
    test = pd.merge(test, group_test, on="client_id", how="left")
    rename_dict = {}
    for col in cols:
        rename_dict[col] = col+"_mean"
    test.rename(rename_dict, axis=1, inplace=True)
    print("Done in {} seconds".format(time.time()-start))

    # max
    print("Max calculating")
    start = time.time()
    group_test = invoice.groupby("client_id")[cols].max()
    test = pd.merge(test, group_test, on="client_id", how="left")

    rename_dict = {}
    for col in cols:
        rename_dict[col] = col+"_max"
    test.rename(rename_dict, axis=1, inplace=True)
    print("Done in {} seconds".format(time.time()-start))

    # min
    print("Min calculating")
    start = time.time()
    group_test = invoice.groupby("client_id")[cols].min()
    test = pd.merge(test, group_test, on="client_id", how="left")

    rename_dict = {}
    for col in cols:
        rename_dict[col] = col+"_min"
    test.rename(rename_dict, axis=1, inplace=True)
    print("Done in {} seconds".format(time.time()-start))

    # std
    print("STD calculating")
    start = time.time()
    group_test = invoice.groupby("client_id")[cols].std()
    test = pd.merge(test, group_test, on="client_id", how="left")

    rename_dict = {}
    for col in cols:
        rename_dict[col] = col+"_std"
    test.rename(rename_dict, axis=1, inplace=True)
    print("Done in {} seconds".format(time.time()-start))

    # skew
    print("Skew calculating")
    start = time.time()
    group_test = invoice.groupby("client_id")[cols].skew()
    test = pd.merge(test, group_test, on="client_id", how="left")

    rename_dict = {}
    for col in cols:
        rename_dict[col] = col+"_skew"
    test.rename(rename_dict, axis=1, inplace=True)
    print("Done in {} seconds".format(time.time()-start))

    # var
    print("Var calculating")
    start = time.time()
    group_test = invoice.groupby("client_id")[cols].var()
    test = pd.merge(test, group_test, on="client_id", how="left")

    rename_dict = {}
    for col in cols:
        rename_dict[col] = col+"_var"
    test.rename(rename_dict, axis=1, inplace=True)
    print("Done in {} seconds".format(time.time()-start))

    # nunique
    print("Nunique calculating")
    start = time.time()
    group_test = invoice.groupby("client_id")[cols].nunique()
    test = pd.merge(test, group_test, on="client_id", how="left")

    rename_dict = {}
    for col in cols:
        rename_dict[col] = col+"_nunique"
    test.rename(rename_dict, axis=1, inplace=True)
    print("Done in {} seconds".format(time.time()-start))

    # Counter Type
    print("Counter type extraction")
    start = time.time()
    cols = ["counter_type"]
    for energy_types in (["ELEC", "GAZ"]):

        group_test = invoice[invoice.counter_type == energy_types].groupby(["client_id"])[
            "counter_type"].count()
        try:
            test = pd.merge(test, group_test, on="client_id", how="left")
        except:
            test = pd.merge(test, group_test, on="client_id", how="left")

        rename_dict = {}
        for col in cols:
            rename_dict[col] = col+"_{}_count".format(energy_types)
        test.rename(rename_dict, axis=1, inplace=True)

    test["isElec"] = (test['counter_type_ELEC_count'] > 0).astype(int)
    print("Done in {} seconds".format(time.time()-start))

    cols = [col for col in test.columns if col not in [
        "target", "client_id", "creation_date"]]

    pred = model.predict_proba(test[cols])[0,1]
    return pred

def file_to_pandas(file):
    file = "\n".join(file.split("\r")).split("\n")
    filtered_file = []
    for i in range(len(file)):
        file[i]= file[i].split(",")
        if file[i] != [""]:
            filtered_file.append(file[i])
    data = pd.DataFrame( data = filtered_file[1:])
    data.columns = filtered_file[0]
    return data

@app.route('/upload',methods=['POST'])
def results():
    
    f1 = request.files['file1']
    fstring1 = f1.read().decode("utf-8")

    f2 = request.files['file2']
    fstring2 = f2.read().decode("utf-8")

    return str(predict(fstring1,fstring2))


if __name__ == "__main__":
    app.run(debug=True)
