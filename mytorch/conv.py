# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np


class Conv1D():
    def __init__(self, in_channel, out_channel, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_size)
        Return:
            out (np.array): (batch_size, out_channel, output_size)
        """
        #print(self.W.shape)
        self.x = x
        batch_size, in_channel, input_size = x.shape
        self.input_size = input_size
        output_size = int(np.floor(((input_size-(self.kernel_size-1)-1)/self.stride)+1))
        out = np.array(np.zeros((batch_size, self.out_channel, int(output_size))))
        for N in range(batch_size):
            for j in range(self.out_channel):
                out_n = self.b[j]
                for k in range(in_channel):
                    temp_out = np.array(np.zeros((output_size)))
                    for i in range(output_size):
                        temp_out[i] = np.correlate(self.W[j][k], x[N][k][i*self.stride:i*self.stride+self.kernel_size])
                    out_n += temp_out
                out[N][j] = out_n
        return out



    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_size)
        Return:
            dx (np.array): (batch_size, in_channel, input_size)
        """
        dx = np.array(np.zeros(self.x.shape))
        batch_size, out_channel, output_size = delta.shape
        for N in range(batch_size):
            for j in range(out_channel):
                for x in range(0, int(np.floor((self.input_size-self.kernel_size)/self.stride))+1):
                    m = x*self.stride
                    for i in range(self.in_channel):
                        for x_p in range(self.kernel_size):
                            dx[N,i,m+x_p] += self.W[j,i,x_p]*delta[N,j,x]
                            self.dW[j,i,x_p] += delta[N,j,x]*self.x[N,i,m+x_p]
        temp_dp = np.sum(delta, axis=0)
        self.db = np.sum(temp_dp, axis=1)

        return dx



class Flatten():
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, in_width)
        Return:
            out (np.array): (batch_size, in_channel * in width)
        """
        self.b, self.c, self.w = x.shape
        out = x.reshape(self.b, self.c * self.w)

        return out

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, in channel * in width)
        Return:
            dx (np.array): (batch size, in channel, in width)
        """
        dx = delta.reshape(self.b, self.c, self.w)
        return dx
