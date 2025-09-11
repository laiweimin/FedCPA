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
from sklearn.preprocessing import LabelEncoder, RobustScaler, OneHotEncoder, OrdinalEncoder, StandardScaler

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


    df = pd.read_csv("./CreditScore/rawdata/credit_score_train.csv")

    # Function to detect outliers
    def detect_outliers_iqr(df):
        outliers = {}

        # Select only numerical columns
        numeric_df = df.select_dtypes(include=['number'])

        for column in numeric_df.columns:
            # Calculate the first (Q1) and third quartiles (Q3)
            Q1 = numeric_df[column].quantile(0.25)
            Q3 = numeric_df[column].quantile(0.75)

            # Calculate the IQR
            IQR = Q3 - Q1

            # Determine the lower and upper bounds
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR

            # Identify outliers
            outlier_mask = (numeric_df[column] < lower_bound) | (numeric_df[column] > upper_bound)
            outliers[column] = numeric_df[column][outlier_mask]

        return outliers

    # Detect outliers
    outlier_results = detect_outliers_iqr(df)

    # List of unique loan values
    unique_loan_types = ['Auto Loan', 'Credit-Builder Loan', 'Debt Consolidation Loan', 'Home Equity Loan',
                         'Mortgage Loan',
                         'No Loan', 'Not Specified', 'Payday Loan', 'Personal Loan', 'Student Loan']

    # Adding a new column for each unique loan type and checking how many times it appears
    for loan_type in unique_loan_types:
        # Replacing '-' and spaces with underscores, converting other characters to lowercase
        cleaned_loan_type = loan_type.replace(' ', '_').replace('-', '_').lower()

        # Counting how many times the loan_type value appears in each row
        df[cleaned_loan_type] = df['type_of_loan'].apply(lambda x: x.count(loan_type))
    df = df.drop(["id", "customer_id", "name", "ssn", "type_of_loan"], axis=1)
    payment_mapping = {
        'High_spent_Large_value_payments': 6,
        # Successfully managing large debts provides the most positive contribution to the credit score.
        'High_spent_Medium_value_payments': 5,
        # Medium-value payments with high spending positively impact the credit score.
        'High_spent_Small_value_payments': 4,
        # Small payments can negatively affect the credit score if debts accumulate over time.
        'Low_spent_Large_value_payments': 3,
        # shows quick financial responsibility, positively affecting the credit score.
        'Low_spent_Medium_value_payments': 2,
        # contributes positively to the credit score by demonstrating debt management.
        'Low_spent_Small_value_payments': 1
        # may limit the credit history and provide minimal contribution to the credit score
    }
    df['payment_behaviour'] = df['payment_behaviour'].map(payment_mapping)
    df['payment_behaviour'] = pd.to_numeric(df['payment_behaviour'], downcast='integer')
    # Convert the credit_mix column to numerical values
    df['credit_mix'] = df['credit_mix'].map({'Good': 2, 'Standard': 1, 'Bad': 0})
    df['credit_mix'] = pd.to_numeric(df['credit_mix'], downcast='integer')
    # Convert the payment_of_min_amount column to numerical values
    df['payment_of_min_amount'] = df['payment_of_min_amount'].map({'Yes': 1, 'No': 0})
    df['payment_of_min_amount'] = pd.to_numeric(df['payment_of_min_amount'], downcast='integer')
    df = pd.get_dummies(df, columns=['occupation'])
    month_map = {
        'January': 1,
        'February': 2,
        'March': 3,
        'April': 4,
        'May': 5,
        'June': 6,
        'July': 7,
        'August': 8
    }
    # Mapping
    df['month'] = df['month'].map(month_map)
    df['month'] = pd.to_numeric(df['month'], downcast='integer')
    # Separate features and target variable
    X = df.drop("credit_score", axis=1)
    y = df.credit_score
    # Columns to apply RobustScaler
    robust_columns = ['total_emi_per_month', 'amount_invested_monthly', 'monthly_balance', "annual_income",
                      "monthly_inhand_salary"]

    # Columns to apply StandardScaler (All columns except robust columns)
    standard_columns = [col for col in X.columns if col not in robust_columns]
    # Create the ColumnTransformer
    scaler = ColumnTransformer(
        transformers=[
            ('standard', StandardScaler(), standard_columns),
            ('robust', RobustScaler(), robust_columns)])

    # Apply to training data
    X_scaled = scaler.fit_transform(X)
    y = np.array(y)

    dataset_image = X_scaled
    dataset_label = y

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