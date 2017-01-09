""" Methods for doing logistic regression."""

import numpy as np
import math as math
from utils import sigmoid
import time


def logistic_predict(weights, data):
    """
    Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to the bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
    Outputs:
        y:          :N x 1 vector of probabilities of being second class. This is the output of the classifier.
    """
    # TODO: Finish this function
    #Add Column of ones to end of data
    N, M = data.shape
    bias_column = np.ones((N,1))
    data_mod = np.hstack((data,bias_column))


    y = sigmoid(np.dot(data_mod,weights))



    return y


def evaluate(targets, y):
    """
    Compute evaluation metrics.
    Inputs:
        targets : N x 1 vector of targets.
        y       : N x 1 vector of probabilities.
    Outputs:
        ce           : (scalar) Cross entropy. CE(p, q) = E_p[-log q]. Here we want to compute CE(targets, y)
        frac_correct : (scalar) Fraction of inputs classified correctly.
    """
    # TODO: Finish this function
    log_y = np.log(y)
    log_y2 = np.log(1-y)
    correct_predictions = 0
    total_predictions = 0
    for index, value in enumerate(y):
        if value >= 0.5 and targets[index] == 1 or value < 0.5 and targets[index] == 0:
            correct_predictions += 1
            total_predictions += 1
        else:
            total_predictions += 1

    ce = -1*np.dot(targets.T, log_y) - np.dot((1-targets).T, log_y2)
    frac_correct = float(correct_predictions) / total_predictions

    return ce, frac_correct


def logistic(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:       The sum of the loss over all data points. This is the objective that we want to minimize.
        df:      (M+1) x 1 vector of accumulative derivative of f w.r.t. weights, i.e. don't need to average over number of sample
        y:       N x 1 vector of probabilities.
    """

    y = logistic_predict(weights, data)

    if hyperparameters['weight_regularization'] is True:
        f, df = logistic_pen(weights, data, targets, hyperparameters)
    else:
        # TODO: compute f and df without regularization
        N, M = data.shape

        #Cost Function
        bias_column = np.ones((N, 1))
        data_mod = np.hstack((data, bias_column))

        #Objective Function
        f = -1*np.sum(targets * np.log(y) + (1-targets)*np.log(1-y))
        #Gradients
        df = np.dot(data_mod.T,(y-targets))



    return f, df, y


def logistic_pen(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:             The sum of the loss over all data points. This is the objective that we want to minimize.
        df:            (M+1) x 1 vector of accumulative derivative of f w.r.t. weights, i.e. don't need to average over number of sample
    """


    N, M = data.shape
    y = logistic_predict(weights, data)

    bias_column = np.ones((N,1))
    data_mod = np.hstack((data,bias_column))
    #Note: I originally removed the bias weight from the weigths, as is typically done in regularization, but the check_grad didn't work if I did that.
    f = -1*np.sum(targets * np.log(y) + (1-targets)*np.log(1-y)) + (hyperparameters['weight_decay']/2.0)*np.dot(weights.T, weights) + math.log(math.sqrt((2*math.pi)/(hyperparameters['weight_decay'])))
    # weights[-1] = 0
    df = np.dot(data_mod.T, (y-targets)) + (hyperparameters['weight_decay'])*weights


    return f, df
