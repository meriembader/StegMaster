import warnings

import pandas as pd
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")


if __name__=="__main__":
    print("Loading Data in the memory...")
    client = pd.read_csv("../data/client.csv")
    print("Loading...")
    invoice = pd.read_csv("../data/invoice.csv")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1111)
    print("Spliting the data...")
    for index_train, index_val in skf.split(client, client.target):
        train = client.loc[index_train].reset_index(drop=True)
        val = client.loc[index_val].reset_index(drop=True)
        break

    train.to_csv("../data/client_train.csv", index=False)
    val.to_csv("../data/client_validation.csv", index=False)
    print("Data is saved")
