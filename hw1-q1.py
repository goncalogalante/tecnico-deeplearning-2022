#!/usr/bin/env python

# Deep Learning Homework 1

import argparse
import random
import os

import numpy as np
import matplotlib.pyplot as plt

import utils

# Added functions

def ReLu(value): 
    return np.maximum(value, 0)

def Softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

def ReLu_derivate(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x

# ------ X -------

def configure_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        y_hat = self.W.dot(x_i)
        index = np.argmax(y_hat)
        if index != y_i:
            self.W[y_i,:] += x_i
            self.W[index,:] -= x_i   
        # Q1.1a


class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        y_one_hot = np.zeros((self.W.shape[0]))
        y_one_hot[y_i] = 1
        
        y_pred = Softmax(self.W.dot(x_i))
        grad =  np.outer(y_pred - y_one_hot, x_i)

        self.W = self.W - learning_rate * grad
        # Q1.1b


class MLP(object):
    # Q3.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size, opt_layers):
        # Initialize an MLP with a single hidden layer.
        
        # Weights initialization 
        self.hiddenl_W = np.random.normal(loc = 0.1,scale= 0.1, size = (hidden_size, n_features))
        self.outl_W = np.random.normal(loc = 0.1, scale = 0.1, size = (n_classes, hidden_size))
    
        # Biases initialization    
        self.hiddenl_b = np.zeros(hidden_size)
        self.outl_b = np.zeros(n_classes)
    

    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.
        
        y_pred = np.zeros(X.shape[0])
        
        for i in range(X.shape[0]):
            x_i = X[i,:].T
            Hl_relu = ReLu(self.hiddenl_W.dot(x_i) + self.hiddenl_b)
            output = Softmax(self.outl_W.dot(Hl_relu) + self.outl_b)
            y_pred[i] = np.argmax(output) 

        return y_pred

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


    def train_epoch(self, X, y, learning_rate=0.001):
        
        for x_i, y_i_aux in zip(X, y):
            x_i = np.asarray(x_i)
            y_i_aux = np.asarray(y_i_aux)
            y_i = np.zeros(self.outl_b.shape[0])
            y_i[y_i_aux]=1
                
            # predict 
            h = ReLu(self.hiddenl_W.dot(x_i) + self.hiddenl_b)
            output = Softmax(self.outl_W.dot(h) + self.outl_b)   
                
            # back propagation output layer
            grad_output_layer = -y_i + output
            grad_outl_w = grad_output_layer[:,None].dot(h[None,:]) # dL/dW (output)
            grad_outl_b = grad_output_layer # dL/db (output)

            
            # back propagation hidden layer
            grad_hidden_layer = self.outl_W.T.dot(grad_output_layer)
            grad_relu_loss = ReLu_derivate(h)*grad_hidden_layer
            grad_hiddenl_w = grad_relu_loss[:, None].dot(x_i[None,:]) # dL/dW (hidden)
            grad_hiddenl_b = grad_relu_loss # dL/db (hidden)
            
            # update weights and biases
            self.outl_W = self.outl_W - learning_rate * grad_outl_w
            self.outl_b = self.outl_b - learning_rate * grad_outl_b
            self.hiddenl_W = self.hiddenl_W - learning_rate * grad_hiddenl_w
            self.hiddenl_b = self.hiddenl_b - learning_rate * grad_hiddenl_b
            
        
        #raise NotImplementedError


def plot(epochs, valid_accs, test_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)
    plt.plot(epochs, valid_accs, label='validation')
    plt.plot(epochs, test_accs, label='test')
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-layers', type=int, default=1,
                        help="""Number of hidden layers (needed only for MLP,
                        not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_classification_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]

    n_classes = np.unique(train_y).size  # 10
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size, opt.layers)
    epochs = np.arange(1, opt.epochs + 1)
    valid_accs = []
    test_accs = []
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        model.train_epoch(
            train_X,
            train_y,
            learning_rate=opt.learning_rate
        )
        valid_accs.append(model.evaluate(dev_X, dev_y))
        test_accs.append(model.evaluate(test_X, test_y))

    # plot
    plot(epochs, valid_accs, test_accs)


if __name__ == '__main__':
    main()
