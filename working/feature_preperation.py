import datetime
import os
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    data_dir = "../data/"
    train = pd.read_csv(os.path.join(data_dir, "client_train.csv"))
    validation = pd.read_csv(os.path.join(data_dir, "client_validation.csv"))
    invoice = pd.read_csv(os.path.join(data_dir, "invoice.csv"))

    invoice["diff_index"] = invoice["new_index"]-invoice["old_index"]
    invoice["ration_diff_index"] = (
        invoice["new_index"]-invoice["old_index"])/invoice["months_number"].apply(lambda x: max(x, 36))

    for df in [train, validation]:
        df.creation_date = pd.to_datetime(df.creation_date)
        df['year'] = df['creation_date'].dt.year
        df['month'] = df['creation_date'].dt.month
        df['dow'] = df['creation_date'].dt.dayofweek
        df['day'] = df['creation_date'].dt.day
        df['doy'] = df['creation_date'].dt.dayofyear
        df['woy'] = df['creation_date'].dt.weekofyear
        df['mend'] = df['creation_date'].dt.is_month_end
        df['mstart'] = df['creation_date'].dt.is_month_start
        df['quarter'] = df['creation_date'].dt.quarter
        df['quartstart'] = df['creation_date'].dt.is_quarter_start
        df['quartend'] = df['creation_date'].dt.is_quarter_end

    cols = ['tarif_type', 'counter_number', 'counter_code', 'reading_remarque',
            'counter_coefficient', 'consommation_level_1', 'consommation_level_2',
            'consommation_level_3', 'consommation_level_4', 'old_index',
            'new_index', 'months_number', 'diff_index',
            'ration_diff_index']

    # mean
    print("Mean calculating")
    start = time.time()
    group_train = invoice.groupby("client_id")[cols].mean()
    group_validation = invoice.groupby("client_id")[cols].mean()
    train = pd.merge(train, group_train, on="client_id", how="left")
    validation = pd.merge(validation, group_validation,
                          on="client_id", how="left")
    rename_dict = {}
    for col in cols:
        rename_dict[col] = col+"_mean"
    train.rename(rename_dict, axis=1, inplace=True)
    validation.rename(rename_dict, axis=1, inplace=True)
    print("Done in {} seconds".format(time.time()-start))

    # max
    print("Max calculating")
    start = time.time()
    group_train = invoice.groupby("client_id")[cols].max()
    group_validation = invoice.groupby("client_id")[cols].max()
    train = pd.merge(train, group_train, on="client_id", how="left")
    validation = pd.merge(validation, group_validation,
                          on="client_id", how="left")

    rename_dict = {}
    for col in cols:
        rename_dict[col] = col+"_max"
    train.rename(rename_dict, axis=1, inplace=True)
    validation.rename(rename_dict, axis=1, inplace=True)
    print("Done in {} seconds".format(time.time()-start))

    # min
    print("Min calculating")
    start = time.time()
    group_train = invoice.groupby("client_id")[cols].min()
    group_validation = invoice.groupby("client_id")[cols].min()
    train = pd.merge(train, group_train, on="client_id", how="left")
    validation = pd.merge(validation, group_validation,
                          on="client_id", how="left")

    rename_dict = {}
    for col in cols:
        rename_dict[col] = col+"_min"
    train.rename(rename_dict, axis=1, inplace=True)
    validation.rename(rename_dict, axis=1, inplace=True)
    print("Done in {} seconds".format(time.time()-start))

    # std
    print("STD calculating")
    start = time.time()
    group_train = invoice.groupby("client_id")[cols].std()
    group_validation = invoice.groupby("client_id")[cols].std()
    train = pd.merge(train, group_train, on="client_id", how="left")
    validation = pd.merge(validation, group_validation,
                          on="client_id", how="left")

    rename_dict = {}
    for col in cols:
        rename_dict[col] = col+"_std"
    train.rename(rename_dict, axis=1, inplace=True)
    validation.rename(rename_dict, axis=1, inplace=True)
    print("Done in {} seconds".format(time.time()-start))

    # skew
    print("Skew calculating")
    start = time.time()
    group_train = invoice.groupby("client_id")[cols].skew()
    group_validation = invoice.groupby("client_id")[cols].skew()
    train = pd.merge(train, group_train, on="client_id", how="left")
    validation = pd.merge(validation, group_validation,
                          on="client_id", how="left")

    rename_dict = {}
    for col in cols:
        rename_dict[col] = col+"_skew"
    train.rename(rename_dict, axis=1, inplace=True)
    validation.rename(rename_dict, axis=1, inplace=True)
    print("Done in {} seconds".format(time.time()-start))

    # var
    print("Var calculating")
    start = time.time()
    group_train = invoice.groupby("client_id")[cols].var()
    group_validation = invoice.groupby("client_id")[cols].var()
    train = pd.merge(train, group_train, on="client_id", how="left")
    validation = pd.merge(validation, group_validation,
                          on="client_id", how="left")

    rename_dict = {}
    for col in cols:
        rename_dict[col] = col+"_var"
    train.rename(rename_dict, axis=1, inplace=True)
    validation.rename(rename_dict, axis=1, inplace=True)
    print("Done in {} seconds".format(time.time()-start))

    # nunique
    print("Nunique calculating")
    start = time.time()
    group_train = invoice.groupby("client_id")[cols].nunique()
    group_validation = invoice.groupby("client_id")[cols].nunique()
    train = pd.merge(train, group_train, on="client_id", how="left")
    validation = pd.merge(validation, group_validation,
                          on="client_id", how="left")

    rename_dict = {}
    for col in cols:
        rename_dict[col] = col+"_nunique"
    train.rename(rename_dict, axis=1, inplace=True)
    validation.rename(rename_dict, axis=1, inplace=True)
    print("Done in {} seconds".format(time.time()-start))

    # Counter Type
    print("Counter type extraction")
    start = time.time()
    cols = ["counter_type"]
    for energy_types in (["ELEC", "GAZ"]):

        group_train = invoice[invoice.counter_type == energy_types].groupby(["client_id"])[
            "counter_type"].count()
        group_validation = invoice[invoice.counter_type == energy_types].groupby(
            ["client_id"])["counter_type"].count()
        try:
            train = pd.merge(train, group_train, on="client_id", how="left")
            validation = pd.merge(
                validation, group_validation, on="client_id", how="left")
        except:
            train = pd.merge(train, group_train, on="client_id", how="left")
            validation = pd.merge(
                validation, group_validation, on="client_id", how="left")

        rename_dict = {}
        for col in cols:
            rename_dict[col] = col+"_{}_count".format(energy_types)
        train.rename(rename_dict, axis=1, inplace=True)
        validation.rename(rename_dict, axis=1, inplace=True)

    train["isElec"] = (train['counter_type_ELEC_count'] > 0).astype(int)
    validation["isElec"] = (
        validation['counter_type_ELEC_count'] > 0).astype(int)
    print("Done in {} seconds".format(time.time()-start))

    print("Saving Files")
    train.to_csv(os.path.join(data_dir, "train.csv"), index=False)
    validation.to_csv(os.path.join(data_dir, "validation.csv"), index=False)
