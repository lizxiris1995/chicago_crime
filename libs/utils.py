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


def generate_train_test(data):
    # TODO: train test split
    raise NotImplementedError


def train(data):
    # TODO: train model
    raise NotImplementedError


def test(data):
    # TODO: test model
    raise NotImplementedError


def visualize():
    #TODO: visualization
    raise NotImplementedError