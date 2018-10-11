from builtins import range
import numpy as np

def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine function.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: a numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: a numpy array of weights, of shape (D, M)
    - b: a numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    """

    num_train = x.shape[0]
    num_dim = np.prod(x.shape[1:])
    x = x.reshape(num_train, num_dim)
    out = np.dot(x, w) + b
    return out


def affine_backward(dout, x, w, b):
    """
    Computes the backward pass of an affine function.

    Inputs:
    - dout: upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: input data, of shape (N, d_1, ... d_k)
      - w: weights, of shape (D, M)
      - b: bias, of shape (M,)

    Returns a tuple of:
    - dx: gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: gradient with respect to w, of shape (D, M)
    - db: gradient with respect to b, of shape (M,)
    """

    dx, dw, db = None, None, None

    num_train = x.shape[0]
    num_dim = np.prod(x.shape[1:])
    x = x.reshape(num_train, num_dim)

    db = np.dot(dout.T, np.ones(num_train))
    dx = np.dot(dout,w.T)
    dw = np.dot(x.T,dout)

    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for rectified linear units (ReLUs).

    Input:
    - x: inputs, of any shape

    Returns a tuple of:
    - out: output, of the same shape as x
    """
    out = np.maximum(0, x)
    
    return out


def relu_backward(dout, x):
    """
    Computes the backward pass for rectified linear units (ReLUs).

    Input:
    - dout: upstream derivatives, of any shape

    Returns:
    - dx: gradient with respect to x
    """
    dx = np.where(x > 0, dout, 0)

    return dx


def softmax_loss(x, y):
    """
    Softmax loss function, vectorized version.
    y_prediction = argmax(softmax(x))
    
    Inputs:
    - X: (float) a tensor of shape (N, #classes)
    - y: (int) ground truth label, a array of length N

    Returns:
    - loss: the cross-entropy loss
    - dx: gradients wrt input x
    """
    # Initialize the loss.
    loss = 0.0
    dx = np.zeros_like(x)

    num_train = x.shape[0]

    scores = x - np.max(x, axis=1, keepdims=True)
    scores_exp = np.exp(scores)
    score_sum = np.sum(scores_exp, axis=1)
    score_sum = (np.ones(scores.shape).T * score_sum).T
    scores = scores_exp / score_sum
    correct_score = scores[range(num_train), y]
    correct_score = np.log(correct_score)
    loss = np.sum(-correct_score)
    loss /= num_train

    dx = np.copy(scores)
    dx[range(num_train), y] -= 1
    dx /= num_train

    return loss, dx