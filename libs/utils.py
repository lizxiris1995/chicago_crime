import os
import os.path
import pandas as pd
import time
import copy
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from .models import fnn


feature_list = ['is_Weekend', 'Holiday', '5d rolling avg', '30 rolling avg',
                'precipitation', 'Average Temp Norm', 'total population',
                'median income', '% 25 older', '% married', '% highschool graduates',
                '% foreign', '% poverty', '% own house', 'rides', 'housing_price_1b',
                'housing_price_2b', 'housing_price_3b']
label = 'Bin'


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.shape[0]

    _, pred = torch.max(output, dim=-1)

    correct = pred.eq(target).sum() * 1.0

    acc = correct / batch_size

    return acc


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


def convert_pandas_to_tensor(model, data):
    """
    Convert pandas dataframe to torch tensor dataset
    :param data: pandas dataframe
    :return: tensor dataset
    """
    features = data[feature_list]
    labels = data[label]

    num_dates = len(set(data['Date']))
    num_wards = len(data['Ward'].unique())
    num_features = len(feature_list)
    if isinstance(model, fnn.FeedforwardNetwork):
        features = features.values.reshape((num_dates*num_wards, num_features))
        labels = labels.values.reshape(num_dates*num_wards, 1)
    else:
        #TODO: elif model is cnn
        features = features.values.reshape((num_dates, 5, 10, num_features))
        labels = labels.values.reshape((num_dates, 5, 10, 1))

    features_tensor = torch.Tensor(features)
    labels_tensor = torch.Tensor(labels)
    dataset = TensorDataset(features_tensor, labels_tensor)

    return dataset


def train_model(model,
                data,
                output_dir,
                batch_size=5,
                shuffle=True,
                learning_rate=0.1,
                reg=0.0005,
                momentum=0.9,
                loss_type='CE',
                save_best=True):
    data['Bin'] = data['Bin'].fillna(1)
    data = data.fillna(0)
    data[feature_list] = (data[feature_list]-data[feature_list].mean())/data[feature_list].std()

    if loss_type == "CE":
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f'Loss type {loss_type} is currently not supported.')

    optimizer = torch.optim.SGD(model.parameters(),
                                learning_rate,
                                momentum=momentum,
                                weight_decay=reg)

    best = 0.0
    best_cm = None
    best_model = None
    idx_list = train_test_split(data)
    resultL = []
    for idx in range(len(idx_list)):
        ind = idx_list[idx]
        train_data = data[data['quarter_ind'].isin(ind[:2])]
        test_data = data[data['quarter_ind'] == ind[3]]

        training_result = train(model, train_data, batch_size, shuffle, idx+1, optimizer, criterion)

        acc, cm = validate(model, test_data, batch_size, shuffle, idx+1, criterion)

        result = training_result.copy()
        result['Testing Accuracy'] = acc.item()
        resultL.append(result)

        if acc > best:
            best = acc
            best_cm = cm
            best_model = copy.deepcopy(model)
    result = pd.concat(resultL, ignore_index=True)
    result.to_csv(output_dir + f'/{model.model_name}.csv')

    print('Best Prec @1 Acccuracy: {:.4f}'.format(best))
    per_cls_acc = best_cm.diag().detach().numpy().tolist()
    for i, acc_i in enumerate(per_cls_acc):
        print("Accuracy of Class {}: {:.4f}".format(i, acc_i))

    if save_best:
        torch.save(best_model.state_dict(),
                   output_dir + '/' + model.model_name + '.pth')

    return None


def train(model, train_data, batch_size, shuffle, epoch, optimizer, criterion):
    iter_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    data = convert_pandas_to_tensor(model, train_data)
    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    outputL = []
    for idx, (data, target) in enumerate(loader):
        start = time.time()

        out = model(data)
        loss = criterion(out, torch.flatten(target-1).long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_acc = accuracy(out, target)

        losses.update(loss.item(), out.shape[0])
        acc.update(batch_acc, out.shape[0])

        iter_time.update(time.time() - start)
        print(('Epoch: [{0}][{1}/{2}]\t'
               'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t'
               'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
               'Prec @1 {top1.val:.4f} ({top1.avg:.4f})\t').format(
            epoch,
            idx,
            len(loader),
            iter_time=iter_time,
            loss=losses,
            top1=acc))
        outputL.append(pd.DataFrame({'Epoch': [epoch], 'Training Losses': [losses.val], 'Training Accuracy': [acc.val.item()]}))
    output = pd.concat(outputL, ignore_index=True)
    return output


def validate(model, test_data, batch_size, shuffle, epoch, criterion):
    iter_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    num_class = 10
    cm = torch.zeros(num_class, num_class)

    data = convert_pandas_to_tensor(model, test_data)
    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)

    for idx, (data, target) in enumerate(loader):
        start = time.time()

        with torch.no_grad():
            out = model(data)
            loss = criterion(out, torch.flatten(target-1).long())

        batch_acc = accuracy(out, target)

        # update confusion matrix
        _, preds = torch.max(out, 1)
        for t, p in zip(target.view(-1), preds.view(-1)):
            cm[t.long(), p.long()] += 1

        losses.update(loss.item(), out.shape[0])
        acc.update(batch_acc, out.shape[0])

        iter_time.update(time.time() - start)
        if idx % 10 == 0:
            print(('Epoch: [{0}][{1}/{2}]\t'
                   'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t').format(
                       epoch,
                       idx,
                       len(loader),
                       iter_time=iter_time,
                       loss=losses,
                       top1=acc))

    cm = cm / cm.sum(1)
    per_cls_acc = cm.diag().detach().numpy().tolist()
    for i, acc_i in enumerate(per_cls_acc):
        print("Accuracy of Class {}: {:.4f}".format(i, acc_i))

    print("* Prec @1: {top1.avg:.4f}".format(top1=acc))

    return acc.avg, cm


def visualize():
    #TODO: visualization
    raise NotImplementedError