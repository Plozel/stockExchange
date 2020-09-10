import pandas as pd
import csv
import yaml
import matplotlib.pyplot as plt
import seaborn as sns


def load_data():
    # get train data
    train = pd.read_csv("../data/train_class.csv", quoting=csv.QUOTE_NONE, error_bad_lines=False)
    return train

def get_cols_with_no_nans(df, col_type):

    if col_type == 'num':
        predictors = df.select_dtypes(exclude=['object'])
    elif col_type == 'no_num':
        predictors = df.select_dtypes(include=['object'])
    elif col_type == 'all':
        predictors = df
    else:
        print('Error : choose a type (num, no_num, all)')
        return 0

    cols_with_no_nans = []
    for col in predictors.columns:
        if not df[col].isnull().any():
            cols_with_no_nans.append(col)

    return cols_with_no_nans


if __name__ == '__main__':
    config = yaml.safe_load(open("config.yaml"))
    train_data = load_data()
    print(train_data.describe())

    num_cols = get_cols_with_no_nans(train_data, 'num')
    cat_cols = get_cols_with_no_nans(train_data, 'no_num')

    print('Number of numerical columns with no nan values :', len(num_cols))
    print('Number of nun-numerical columns with no nan values :', len(cat_cols))

    train_data["25/75/YE"], _ = pd.factorize(train_data["25/75/YE"])

    # pd.to_datetime(train_data["Date"]).dt.strftime('%Y-%m-%d')
    # scale the data between 0  to 1
    train_data.iloc[:, 2:] = (train_data.iloc[:, 2:] - train_data.iloc[:, 2:].min()) / train_data.iloc[:, 2:].max()
    print(train_data.columns)

    # print(train_data.loc[:, 'class'].value_counts(normalize=True))

    train_data.hist(figsize=(30, 30))

    plt.show()

    C_mat = train_data.corr()
    fig = plt.figure(figsize=(15, 15))

    sns.heatmap(C_mat, vmax=.8, square=True)
    plt.show()

