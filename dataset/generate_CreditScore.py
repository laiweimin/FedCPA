import numpy as np
import os
import sys
import random

import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler, OneHotEncoder, OrdinalEncoder

from utils.dataset_utils import check, separate_data, split_data, save_file
from torch.utils.data import TensorDataset, DataLoader

random.seed(1)
np.random.seed(1)
num_clients = 100
dir_path = "CreditScore/"


# Allocate data to users
def generate_dataset(dir_path, num_clients, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return


    df = pd.read_csv("./CreditScore/rawdata/clean_data.csv")
    train_data = df[df['is_train'] == True].drop(columns=['is_train'])
    X = train_data.drop(columns=['Credit_Score'])
    y = train_data['Credit_Score']

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(y)
    for index, class_name in enumerate(label_encoder.classes_):
        print(f"Class '{class_name}' is encoded as {index}")

    num_cols = X.select_dtypes(exclude=['object']).columns.tolist()
    one_hot_cols = ["Occupation", "Payment_of_Min_Amount"]
    ordinal_cols = ["Credit_Mix", "Spending_Level", "Payment_Value"]
    ordinal_categories = [
        ['Bad', 'Standard', 'Good'],
        ['Low', 'High'],
        ['Small', 'Medium', 'Large']
    ]
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', RobustScaler(), num_cols),
            ('one_hot_enc', OneHotEncoder(handle_unknown='ignore'), one_hot_cols),
            ('ordinal_enc',
             OrdinalEncoder(categories=ordinal_categories, handle_unknown="use_encoded_value", unknown_value=-1),
             ordinal_cols)
        ]
    )
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    features = pipeline.fit_transform(X)

    dataset_image = features
    dataset_label = labels

    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, 
                                    niid, balance, partition, class_per_client=10)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition)


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    generate_dataset(dir_path, num_clients, niid, balance, partition)