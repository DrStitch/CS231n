import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]
  for i in range(num_train):
    f = X[i].dot(W)
    f_s = f - f.max()
    loss += -f_s[y[i]] + np.log(np.exp(f_s).sum())
    #for j in range(num_class):
    #  if j == y[i]:
    #    dW[:, j] -= X[i]
    #  dW[:, j] += np.exp(f_s)[j] * X[i] / np.exp(f_s).sum()
    dW[:, y[i]] -= X[i]
    dW += X[i].reshape(-1, 1).dot(np.exp(f_s).reshape(1, -1)) / np.exp(f_s).sum()

              
  loss = loss / num_train + 0.5 * reg * (W*W).sum()
  dW = dW / num_train + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]
  f = X.dot(W)
  f_s = f - f.max()
  loss = -f_s[np.arange(num_train), y].sum() + np.log(np.exp(f_s).sum(1)).sum()
  loss = loss / num_train + 0.5 * reg * np.sum(W * W)
  M = np.zeros((num_train, num_class))
  M[np.arange(num_train), y] = 1
  dW -= X.T.dot(M)
  dW += X.T.dot(np.exp(f_s) / np.exp(f_s).sum(1, keepdims=True))
  dW = dW / num_train + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

