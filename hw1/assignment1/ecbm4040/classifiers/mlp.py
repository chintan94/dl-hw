from builtins import range
from builtins import object
import numpy as np

from ecbm4040.layer_funcs import *
from ecbm4040.layer_utils import *

class MLP(object):
    """
    MLP with an arbitrary number of dense hidden layers,
    and a softmax loss function. For a network with L layers,
    the architecture will be

    input >> DenseLayer x (L - 1) >> AffineLayer >> softmax_loss >> output

    Here "x (L - 1)" indicate to repeat L - 1 times. 
    """
    def __init__(self, input_dim=3072, hidden_dims=[200,200], num_classes=10, reg=0.0, weight_scale=1e-2):
        """
        Inputs:
        - reg: (float) L2 regularization
        - weight_scale: (float) for layer weight initialization
        """
        self.num_layers = len(hidden_dims) + 1
        self.reg = reg
        self.velocity = [0,0,0,0,0,0]
        dims = [input_dim] + hidden_dims
        layers = []
        for i in range(len(dims)-1):
            layers.append(DenseLayer(input_dim=dims[i], output_dim=dims[i+1], weight_scale=weight_scale))
        layers.append(AffineLayer(input_dim=dims[-1], output_dim=num_classes, weight_scale=weight_scale))
        
        self.layers = layers

    def loss(self, X, y):
        """
        Calculate the cross-entropy loss and then use backpropogation
        to get gradients wst W,b in each layer.
        
        Inputs:
        - X: input data
        - y: ground truth
        
        Return loss value(float)
        """
        loss = 0.0
        reg = self.reg
        num_layers = self.num_layers
        layers = self.layers
        ###################################################
        #TODO: Feedforward                                #
        ###################################################
        out1 = self.layers[0].feedforward(X)
        out2 = self.layers[1].feedforward(out1)
        out3 = self.layers[2].feedforward(out2)

        # print(out2.shape, y.shape)

        losss, dsoft = softmax_loss(out3, y)
        self.dsoft = dsoft
        loss += losss
        
        ###################################################
        #TODO: Backpropogation                            #
        ###################################################
        dout3 = self.layers[2].backward(self.dsoft)
        dout2 = self.layers[1].backward(dout3)
        dout1 = self.layers[0].backward(dout2)
            
        ###################################################
        # TODO: Add L2 regularization                     #
        ###################################################
        square_weights = np.sum(np.power(self.layers[0].params[0],2)) + np.sum(np.power(self.layers[1].params[0],2)) + np.sum(np.power(self.layers[2].params[0],2))
        loss += 0.5 * reg * square_weights        
        
        ###################################################
        #              END OF YOUR CODE                   #
        ###################################################
        
        return loss

    def step(self, learning_rate=1e-5):
        """
        Use SGD to implement a single-step update to each weight and bias.
        """
        ###################################################
        #TODO: Use SGD  to update variables in layers     #
        ###################################################


        layer1, layer2, layer3 = self.layers[0], self.layers[1], self.layers[2]

        params = layer1.params + layer2.params + layer3.params
        grads = layer1.gradients + layer2.gradients + layer3.gradients  

        # Add L2 regularization
        reg = self.reg
        grads = [grad + reg * params[i] for i, grad in enumerate(grads)]

        self.velocity = [item[0] * 0.8 + learning_rate * item[1] for item in zip(self.velocity, grads)]
        param = [item[0] - item[1] for item in zip(params, self.velocity)]
        params = param

        
        ###################################################
        #              END OF YOUR CODE                   #
        ###################################################
        num_layers = self.num_layers
        # update parameters in layers
        for i in range(num_layers):
            self.layers[i].update_layer(params[2*i:2*(i+1)])
        

    def predict(self, X):
        """
        Return the label prediction of input data
        
        Inputs:
        - X: (float) a tensor of shape (N, D)
        
        Returns: 
        - predictions: (int) an array of length N
        """
        predictions = None
        num_layers = self.num_layers
        layers = self.layers
        #######################################################
        #TODO: Remember to use functions in class SoftmaxLayer#
        #######################################################

        out1 = self.layers[0].feedforward(X)
        out2 = self.layers[1].feedforward(out1)
        out3 = self.layers[2].feedforward(out2)

        predictions = np.argmax(out3, axis=1)

        
        #######################################################
        #                 END OF YOUR CODE                    #
        #######################################################
        
        return predictions
    
    def check_accuracy(self, X, y):
        """
        Return the classification accuracy of input data
        
        Inputs:
        - X: (float) a tensor of shape (N, D)
        - y: (int) an array of length N. ground truth label 
        Returns: 
        - acc: (float) between 0 and 1
        """
        y_pred = self.predict(X)
        acc = np.mean(np.equal(y, y_pred))
        
        return acc
        
        


