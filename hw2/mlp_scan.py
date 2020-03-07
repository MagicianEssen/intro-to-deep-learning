# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import os
import sys

sys.path.append('mytorch')
from loss import *
from activation import *
from linear import *
from conv import *


class CNN_SimpleScanningMLP():
    def __init__(self):
        ## Your code goes here -->
        # self.conv1 = ???
        # self.conv2 = ???
        # self.conv3 = ???
        # ...
        # <---------------------
        self.conv1 = Conv1D(24, 8, 8, 4)
        self.conv2 = Conv1D(8, 16, 1, 1)
        self.conv3 = Conv1D(16, 4, 1, 1)
        self.flatten = Flatten()
        self.layers = [self.conv1, ReLU(), self.conv2, ReLU(),self.conv3, self.flatten]

    def __call__(self, x):
        # Do not modify this method
        return self.forward(x)

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        # w1, w2, w3 contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN

        w1,w2,w3 = weights
        #print(w1.shape, w2.shape, w3.shape)
        w1 = w1.T
        new_w1 = np.zeros((8,8,24))
        for i in range(len(w1)):
            new_w1[i,:,:] = w1[i,:].reshape(8,24)
        final_w1 = np.zeros((8,24,8))
        for j in range(len(new_w1)):
            final_w1[j,:,:] = new_w1[j,:,:].T

        w2 = w2.T
        new_w2 = np.array(np.zeros((16,1,8)))
        for i in range(len(w2)):
            new_w2[i,:,:] = w2[i,:].reshape(1,8)
        final_w2 = np.array(np.zeros((16,8,1)))
        for j in range(len(new_w2)):
            final_w2[j,:,:] = new_w2[j,:,:].T

        w3 = w3.T
        new_w3 = np.array(np.zeros((4,1,16)))
        for i in range(len(w3)):
            new_w3[i,:,:] = w3[i,:].reshape(1,16)
        final_w3 = np.array(np.zeros((4,16,1)))
        for j in range(len(new_w3)):
            final_w3[j,:,:] = new_w3[j,:,:].T

        self.conv1.W = final_w1
        self.conv2.W = final_w2
        self.conv3.W = final_w3

        # self.layers[0].W = final_w1
        # self.layers[2].W = final_w2
        # self.layers[4].W = final_w3

    def forward(self, x):
        """
        Do not modify this method

        Argument:
            x (np.array): (batch size, in channel, in width)
        Return:
            out (np.array): (batch size, out channel , out width)
        """
        #self.batch_size, self.in_channel, self.in_width = x.shape
        out = x
        for layer in self.layers:
            #print(layer)
            out = layer(out)
            #print(out.shape)
        return out

    def backward(self, delta):
        """
        Do not modify this method

        Argument:
            delta (np.array): (batch size, out channel, out width)
        Return:
            dx (np.array): (batch size, in channel, in width)
        """

        for layer in self.layers[::-1]:
            delta = layer.backward(delta)
        return delta


class CNN_DistributedScanningMLP():
    def __init__(self):
        ## Your code goes here -->
        # self.conv1 = ???
        # self.conv2 = ???
        # self.conv3 = ???
        # ...
        # <---------------------
        self.conv1 = Conv1D(24, 2, 2, 2)
        self.conv2 = Conv1D(2, 8, 2, 2)
        self.conv3 = Conv1D(8, 4, 2, 1)
        self.flatten = Flatten()
        self.layers = [self.conv1, ReLU(), self.conv2, ReLU(),self.conv3, self.flatten]

    def __call__(self, x):
        # Do not modify this method
        return self.forward(x)

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        # w1, w2, w3 contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN

        w1, w2, w3 = weights
        #print(w1.shape,w2.shape,w3.shape)
        w1 = w1[:48,:2].T
        new_w1 = np.zeros((2,2,24))
        for i in range(len(w1)):
            new_w1[i,:,:] = w1[i,:].reshape(2,24)
        final_w1 = np.zeros((2,24,2))
        for j in range(len(new_w1)):
            final_w1[j,:,:] = new_w1[j,:,:].T

        w2 = w2[:4,:8].T
        new_w2 = np.array(np.zeros((8,2,2)))
        for i in range(len(w2)):
            new_w2[i,:,:] = w2[i,:].reshape(2,2)
        final_w2 = np.array(np.zeros((8,2,2)))
        for j in range(len(new_w2)):
            final_w2[j,:,:] = new_w2[j,:,:].T

        w3 = w3[:16,:4].T
        new_w3 = np.array(np.zeros((4,2,8)))
        for i in range(len(w3)):
            new_w3[i,:,:] = w3[i,:].reshape(2,8)
        final_w3 = np.array(np.zeros((4,8,2)))
        for j in range(len(new_w3)):
            final_w3[j,:,:] = new_w3[j,:,:].T


        self.conv1.W = final_w1
        self.conv2.W = final_w2
        self.conv3.W = final_w3


    def forward(self, x):
        """
        Do not modify this method

        Argument:
            x (np.array): (batch size, in channel, in width)
        Return:
            out (np.array): (batch size, out channel , out width)
        """

        out = x
        for layer in self.layers:
            #print(layer)
            out = layer(out)
        return out

    def backward(self, delta):
        """
        Do not modify this method

        Argument:
            delta (np.array): (batch size, out channel, out width)
        Return:
            dx (np.array): (batch size, in channel, in width)
        """

        for layer in self.layers[::-1]:
            delta = layer.backward(delta)
        return delta
