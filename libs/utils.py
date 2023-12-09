import os
import os.path
import pandas as pd
import numpy as np
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from .models import fnn, cnn, rnn


feature_list = ['is_Weekend', 'Holiday', '5d rolling avg', '30 rolling avg',
                'precipitation', 'Average Temp Norm', 'total population',
                'median income', '% 25 older', '% married', '% highschool graduates',
                '% foreign', '% poverty', '% own house', 'rides', 'housing_price_1b',
                'housing_price_2b', 'housing_price_3b']
label = 'Bin'
num_wards = 50


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


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        """
        Implement forward of focal loss
        :param input: input predictions
        :param target: labels
        :return: tensor of focal loss in scalar
        """

        z_i = F.cross_entropy(input=input, target=target, weight=self.weight.float(), reduction='none')
        p_t = torch.exp(-z_i) / torch.sum(torch.exp(-z_i))
        loss = ((1 - p_t) ** self.gamma * z_i).mean()

        return loss


def accuracy(output, target, model):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.shape[0]
    if isinstance(model, rnn.RecurrentNeuralNetwork):
        batch_size = batch_size*num_wards

    _, pred = torch.max(output, dim=-1)

    target = torch.flatten(target)
    correct = pred.eq(target).sum() * 1.0

    acc = correct / batch_size

    return acc


def reweight(cls_num_list, beta=0.9999):
    """
    Implement reweighting by effective numbers
    :param cls_num_list: a list containing # of samples of each class
    :param beta: hyper-parameter for reweighting, see paper for more details
    :return:
    """

    N = np.array(cls_num_list)
    factor = (1 - beta) / (1 - np.power(beta, N))
    per_cls_weights = torch.tensor(factor / np.sum(factor) * len(N))

    return per_cls_weights


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


def regenerate_label(data):
    """
    regenerate the label based on total number of crimes across all wards
    bucket counts into bins
    :param data: pandas dataframe
    :return labels dataframe that has column: date, total counts and bins
    """
    #TODO: make the bins resonable for other crime types
    labels_df = data.groupby('Date')['Counts'].sum()
    labels_df = pd.DataFrame(labels_df.shift(-1, fill_value=0)).reset_index()

    bins = [-np.inf, 50, 100, 150, 200, 250, 300, np.inf]
    bins_label = [0, 1, 2, 3, 4, 5, 6]
    num_bins = len(bins_label)
    labels_df['Bin'] = pd.cut(labels_df['Counts'], bins=bins, labels=bins_label, right=False)
    return labels_df, num_bins


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
    elif isinstance(model, cnn.ConvolutionalNetwork):
        features = features.values.reshape((num_dates, 5, 10, num_features))
        # channel should be on the second dimension
        features = np.transpose(features, (0, 3, 1, 2))
        labels = labels.values.reshape(num_dates, num_wards)
    elif isinstance(model, rnn.RecurrentNeuralNetwork):
        features = features.values.reshape(num_dates, num_wards, num_features)
        labels = labels.values.reshape(num_dates, num_wards)

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
                save_best=True,
                beta=0.999):
    data['Bin'] = data['Bin'].fillna(1)
    data = data.fillna(0)
    # data[feature_list] = (data[feature_list]-data[feature_list].mean())/data[feature_list].std()

    if loss_type == "CE":
        criterion = nn.CrossEntropyLoss()
    elif loss_type == 'Focal':
        cls_num_list = data.groupby('Bin')['Bin'].count().to_list()
        per_cls_weights = reweight(cls_num_list, beta=beta)
        criterion = FocalLoss(weight=per_cls_weights, gamma=1)
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
    outputL = []
    for idx in range(len(idx_list)):
        ind = idx_list[idx]
        train_data = data[data['quarter_ind'].isin(ind[:2])]
        test_data = data[data['quarter_ind'] == ind[3]]

        training_result = train(model, train_data, batch_size, shuffle, idx+1, optimizer, criterion)

        acc, cm, data_w_pred = validate(model, test_data, batch_size, shuffle, idx+1, criterion)
        outputL.append(data_w_pred)

        result = training_result.copy()
        result['Testing Accuracy'] = acc.item()
        resultL.append(result)

        if acc > best:
            best = acc
            best_cm = cm
            best_model = copy.deepcopy(model)
    result = pd.concat(resultL, ignore_index=True)
    result.to_csv(output_dir + f'/{model.model_name}_accy.csv')

    outputL[-1].to_csv(output_dir + f'/{model.model_name}_pred.csv')

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
        target = target - 1
        start = time.time()

        out = model(data)
        if isinstance(model, cnn.ConvolutionalNetwork):
            target = torch.reshape(target, (-1, 1))
        loss = criterion(out, torch.flatten(target).long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_acc = accuracy(out, target, model)

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

    outputL = []
    for idx, (data, target) in enumerate(loader):
        target = target - 1
        if isinstance(model, cnn.ConvolutionalNetwork):
            target = torch.reshape(target, (-1, 1))
        start = time.time()

        with torch.no_grad():
            out = model(data)
            loss = criterion(out, torch.flatten(target).long())

        outputL.append(pd.DataFrame(out).idxmax(axis=1))
        batch_acc = accuracy(out, target, model)

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

    output = pd.concat(outputL, ignore_index=True)
    test_data['Pred'] = list(output)
    return acc.avg, cm, test_data


def visualize():
    #TODO: visualization
    raise NotImplementedError