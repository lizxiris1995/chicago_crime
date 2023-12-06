import os
import os.path
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader


feature_list = ['is_Weekend', 'Holiday', '5d rolling avg', '30 rolling avg',
                'precipitation', 'Average Temp Norm', 'total population',
                'median income', '% 25 older', '% married', '% highschool graduates',
                '% foreign', '% poverty', '% own house', 'rides', 'housing_price_1b',
                'housing_price_2b', 'housing_price_3b']
label = 'Bin'


def create_missing_dirs(path):

    if isinstance(path, list):
        for dir_ in path:
            create_missing_dirs(dir_)
    else:
        if path[-1] == '/':
            pass
        else:
            tail = path[path.rfind('/'):]
            if '.' not in tail:
                path = path + '/'

        dir_ = os.path.dirname(path)
        if not os.path.exists(dir_):
            try:
                os.makedirs(dir_)
            except FileExistsError:
                pass


def add_quarter_index(data):
    """
    :param data: pandas dataframe
    :return: dataframe with quarter index added
    """
    data['quarter_ind'] = ((pd.to_datetime(data.Date).dt.month
                         + (pd.to_datetime(data.Date).dt.year-2011)*12-1)//3+1)
    return data


def train_test_split(data):
    m = min(data['quarter_ind'])
    n = max(data['quarter_ind'])
    idx_seq = sorted(list(data['quarter_ind'].unique()))
    idx_list = []
    for i in range(m-1, n-3):
        idx_list.append(idx_seq[i: i + 4])
    return idx_list


def convert_pandas_to_tensor(data):
    """
    Convert pandas dataframe to torch tensor dataset
    :param data: pandas dataframe
    :return: tensor dataset
    """
    features = data[feature_list]
    labels = data[label]

    num_dates = len(set(data['Date']))
    features = features.reshape((num_dates, 5, 10, len(feature_list)))
    labels = labels.reshape((num_dates, 5, 10, 1))

    features_tensor = torch.Tensor(features)
    labels_tensor = torch.Tensor(labels)
    dataset = TensorDataset(features_tensor, labels_tensor)
    return dataset


def train(model, data, batch_size=5, shuffle=True):
    # TODO: train model
    idx_list = train_test_split(data)
    for idx in idx_list:
        train_data = data[data['quarter_ind'].isin(idx[:2])]
        test_data = data[data['quarter_ind'] == idx[3]]
        train_dataset = convert_pandas_to_tensor(train_data)
        test_dataset = convert_pandas_to_tensor(test_data)
        trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    return None


def test(data):
    # TODO: test model
    raise NotImplementedError


def visualize():
    #TODO: visualization
    raise NotImplementedError