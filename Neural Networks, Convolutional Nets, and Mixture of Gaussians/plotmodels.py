#Leo Woiceshyn, 998082159, for CSC2515 Fall 2016
#This script is used to visualize the first layer of weights on a NN and a CNN model
import sys
import matplotlib.pyplot as plt
from util import LoadData, Load, Save, DisplayPlot

def ShowMeans(means, number=0):
  """Show the cluster centers as images."""
  plt.figure(number)
  plt.clf()
  for i in xrange(means.shape[1]):
    plt.subplot(1, means.shape[1], i+1)
    plt.imshow(means[:, i].reshape(48, 48), cmap=plt.cm.gray)
  plt.draw()
  raw_input('Press Enter.')

def ShowMeansCNN(means, number=0):
  """Show the cluster centers as images."""
  plt.figure(number)
  plt.clf()
  for i in xrange(means.shape[3]):
    plt.subplot(1, means.shape[3], i+1)
    plt.imshow(means[:,:,0,i], cmap=plt.cm.gray)
  plt.draw()
  raw_input('Press Enter.')



nn = Load('nn_model.npz')
ShowMeans(nn['W1'], number=0)

cnn = Load('cnn_model.npz')
ShowMeansCNN(cnn['W1'], number=1)
