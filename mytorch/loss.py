# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import os

# The following Criterion class will be used again as the basis for a number
# of loss functions (which are in the form of classes so that they can be
# exchanged easily (it's how PyTorch and other ML libraries do it))

class Criterion(object):
    """
    Interface for loss functions.
    """

    # Nothing needs done to this class, it's used by the following Criterion classes

    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None
        self.storex = None
        self.storey = None

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented

class SoftmaxCrossEntropy(Criterion):
    """
    Softmax loss
    """

    def __init__(self):
        super(SoftmaxCrossEntropy, self).__init__()

    def forward(self, x, y):
        """
        Argument:
            x (np.array): (batch size, 10)
            y (np.array): (batch size, 10)
        Return:
            out (np.array): (batch size, )
        """
        self.logits = x
        self.labels = y
        self.storex = np.copy(x)
        self.storey = np.copy(y)
        self.loss = []
        self.loss = (-1.0) * (y * np.log(np.exp(x)/np.exp(x).sum(axis = 1, keepdims = True))).sum(axis = 1)
        #for i in range(len(x)):
        #    row_sum = np.sum(np.exp(x[i]))
        #    idx = np.where(y[i] == 1)
        #    self.loss.append((-np.log((np.exp(x[i][idx])/row_sum)))[0])
        #self.loss = np.asarray(self.loss)
        #np.squeeze(self.loss,axis=0)
        return self.loss

    def derivative(self):
        """
        Return:
            out (np.array): (batch size, 10)
        """
        x= self.storex
        y= self.storey
        output = []
        expo = np.exp(x)
        sum_expo = []
        for i in range(len(expo)):
            sum_expo.append(sum(expo[i]))

        for i in range(len(expo)):
            expo[i] = expo[i]/sum_expo[i]

        return expo-y
