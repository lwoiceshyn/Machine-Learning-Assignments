# Leo Woiceshyn, Student Number 998082159, for CSC2515 Assignment 1

import numpy as np
from l2_distance import l2_distance

def run_knn(k, train_data, train_labels, valid_data):
    """Uses the supplied training inputs and labels to make
    predictions for validation data using the K-nearest neighbours
    algorithm.

    Note: N_TRAIN is the number of training examples,
          N_VALID is the number of validation examples,
          and M is the number of features per example.

    Inputs:
        k:            The number of neighbours to use for classification
                      of a validation example.
        train_data:   The N_TRAIN x M array of training
                      data.
        train_labels: The N_TRAIN x 1 vector of training labels
                      corresponding to the examples in train_data
                      (must be binary).
        valid_data:   The N_VALID x M array of data to
                      predict classes for.

    Outputs:
        valid_labels: The N_VALID x 1 vector of predicted labels
                      for the validation data.
    """


    # TODO call l2_distance to compute distance between valid data and train data
    # Creates a N_VALID x N_TRAIN matrix with rows representing the validation data examples and columns representing the l2 distance to each training example
    distance = l2_distance(valid_data.T, train_data.T)

    # TODO sort the distance to get top k nearest data

    #For the k=1 case, find the minimum value in each row in the distance matrix, and convert to a 1 x N_VALID matrix of indices
    if k == 1:
        nearest = np.array([], dtype=np.int64)
        for i in xrange(distance.shape[0]):
            min_index = np.argmin(distance[i, :]) #Finds the index of the nearest training example to each validation set example
            nearest = np.append(nearest, min_index)
    # For the k>1 case, create a k x N_VALID matrix where each column is k indices corresponding to the k smallest values in each row of distance
    else:
        nearest = np.zeros((k, distance.shape[0]), dtype=np.int64) #Creates a k x N_VALID matrix of zeros to substitute, with int type since they are indices
        for i in xrange(distance.shape[0]):
            min_indices = np.argpartition(distance[i, :], k)[:k]
            nearest[:, i] = min_indices
        #print nearest

    #Turn labels into 1X200 row vector
    train_labels = train_labels.reshape(-1)

    #Use the indices to find the corresponding training set labels for the k closest training examples
    if k == 1:
        valid_labels = train_labels[nearest]
    else:
        valid_labels = np.zeros((k,nearest.shape[1]), dtype=np.int64) #Creates a k x N_VALID matrix of zeros to substitute, with int type since they are indices
        for i in xrange(k):
            nearest_row = nearest[i, :]
            valid_labels[i, :] = train_labels[nearest_row]




    #print valid_labels
    # note this only works for binary labels
    if k > 1:
        valid_labels = (np.mean(valid_labels, axis=0) >= 0.5).astype(np.int)
        #print valid_labels
    valid_labels = valid_labels.reshape(-1,1)

    return valid_labels
