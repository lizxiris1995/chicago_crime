import os
import os.path


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