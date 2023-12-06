import os
import os.path
import pandas as pd


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


def train(model, data):
    # TODO: train model
    idx_list = train_test_split(data)
    for idx in idx_list:
        train_data = data[data['quarter_ind'].isin(idx[:2])]
        test_data = data[data['quarter_ind'] == idx[3]]
    return None


def test(data):
    # TODO: test model
    raise NotImplementedError


def visualize():
    #TODO: visualization
    raise NotImplementedError