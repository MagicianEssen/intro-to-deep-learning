# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np

class BatchNorm(object):

    def __init__(self, in_feature, alpha=0.9):

        # You shouldn't need to edit anything in init

        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None

        # The following attributes will be tested
        self.var = np.ones((1, in_feature))
        self.mean = np.zeros((1, in_feature))

        self.gamma = np.ones((1, in_feature))
        self.dgamma = np.zeros((1, in_feature))

        self.beta = np.zeros((1, in_feature))
        self.dbeta = np.zeros((1, in_feature))

        # inference parameters
        self.running_mean = np.zeros((None))
        self.running_var = np.ones((None))

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):
        """
        Argument:
            x (np.array): (batch_size, in_feature)
            eval (bool): inference status

        Return:
            out (np.array): (batch_size, in_feature)
        """

        # if eval:
        #    # ???
        if eval:
            self.mean = self.running_mean
            self.var  = self.running_var
            self.norm = (x-self.mean)/((self.var+self.eps)**(0.5))
            self.out = self.gamma * self.norm + self.beta
            return self.out

        self.x = x

        # self.mean = # ???
        # self.var = # ???
        # self.norm = # ???
        # self.out = # ???

        # Update running batch statistics
        # self.running_mean = # ???
        # self.running_var = # ???
        self.norm = np.copy(x)
        #for i in range(len(x)):
        #    mean_i = sum(x[i])/len(x[i])
        #    var_i = 0
        #    for j in range(len(x[i])):
        #        var_i += (x[i][j]-mean_i)**2
        #    var_i = var_i/len(x[i])
        #    self.mean[0][i] = mean_i
        #    self.var[0][i] = var_i
        #    self.norm[i] = (self.norm[i]-mean_i)/((var_i+self.eps)**(1/2))
        #self.out = np.matmul(self.gamma,self.norm) + self.beta
        
        self.mean = np.mean(x,axis=0)
        #print(self.x.shape)
        self.var = np.var(x, axis =0)
        self.norm = (x-self.mean)/((self.var+self.eps)**(0.5))
        self.out = self.gamma * self.norm + self.beta

        self.running_mean = self.alpha * self.running_mean + (1-self.alpha) * self.mean
        self.running_var  = self.alpha * self.running_var  + (1-self.alpha) * self.var

        return self.out


    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, in feature)
        Return:
            out (np.array): (batch size, in feature)
        """
        #print(delta.shape, self.gamma.shape)

        L_xhat = delta * self.gamma
        L_var = (-0.5)*((L_xhat * (self.x-self.mean) * ((self.var+self.eps)**(-1.5))).sum(axis=0))
        L_mean = -((L_xhat * ((self.var+self.eps)**(-0.5)))).sum(axis=0) - (2.0/len(self.x))* L_var * ((self.x-self.mean).sum(axis=0))
        L_x    = (L_xhat * ((self.var+self.eps)**(-0.5))) + (L_var*((2.0/len(self.x))*(self.x-self.mean))) + L_mean*(1.0/(len(self.x)))
        self.dbeta = delta.sum(axis = 0, keepdims=True)
        #print(delta.shape, self.norm.shape)
        self.dgamma = (delta * self.norm).sum(axis = 0, keepdims=True)

        return L_x
