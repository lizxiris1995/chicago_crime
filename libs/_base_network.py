import numpy as np


class _baseNetwork:
    def __init__(self, input_size, num_classes):
        self._input_size = input_size
        self._num_classes = num_classes

        self._weights = dict()
        self._gradients = dict()

    def weight_init(self):
        pass

    def forward(self):
        pass

    def softmax(self, scores):
        """
        Compute softmax scores given the raw output from the model

        :param scores: raw scores from the model (N, num_classes)
        :return:
            prob: softmax probabilities (N, num_classes)
        """
        prob = []

        for score in scores:
            exp_score = [np.exp(s) for s in score]
            prob.append(list(exp_score/sum(exp_score)))

        return prob

    def cross_entropy_loss(self, x_pred, y):
        """
        Compute Cross-Entropy Loss based on prediction of the network and labels
        :param x_pred: Probabilities from the model (N, num_classes)
        :param y: Labels of instances in the batch
        :return: The computed Cross-Entropy Loss
        """
        loss = []

        for i in range(len(x_pred)):
            log_x = np.log(x_pred[i])
            label = [1 if _ == y[i] else 0 for _ in range(len(log_x))]
            loss.append(-sum(label*log_x))

        return np.mean(loss)

    def compute_accuracy(self, x_pred, y):
        """
        Compute the accuracy of current batch
        :param x_pred: Probabilities from the model (N, num_classes)
        :param y: Labels of instances in the batch
        :return: The accuracy of the batch
        """
        acc = []

        for i in range(len(x_pred)):
            pred = np.array(x_pred[i]).argmax()
            acc.append(int(pred == y[i]))

        return np.mean(acc)

    def sigmoid(self, X):
        """
        Compute the sigmoid activation for the input

        :param X: the input data coming out from one layer of the model (N, layer size)
        :return:
            out: the value after the sigmoid activation is applied to the input (N, layer size)
        """
        out = []

        for _ in X:
            out.append([1 / (1 + np.exp(-x)) for x in _])

        return out

    def sigmoid_dev(self, x):
        """
        The analytical derivative of sigmoid function at x
        :param x: Input data
        :return: The derivative of sigmoid function at x
        """
        ds = []

        for _ in x:
            ds.append([np.exp(-a) / (1 + np.exp(-a)) ** 2 for a in _])

        return ds

    def ReLU(self, X):
        """
        Compute the ReLU activation for the input

        :param X: the input data coming out from one layer of the model (N, layer size)
        :return:
            out: the value after the ReLU activation is applied to the input (N, layer size)
        """
        out = []

        for _ in X:
            out.append([max(0, x) for x in _])

        return out

    def ReLU_dev(self, X):
        """
        Compute the gradient ReLU activation for the input

        :param X: the input data coming out from one layer of the model (N, layer size)
        :return:
            out: gradient of ReLU given input X
        """
        out = []

        for _ in X:
            out.append([0 if x < 0 else 1 for x in _])

        return out