#Leo Woiceshyn, 998082159, for CSC2515 Fall 2016
from __future__ import division
from __future__ import print_function

from util import LoadData, Load, Save, DisplayPlot
from conv2d import conv2d as Conv2D
from nn import Affine, ReLU, AffineBackward, ReLUBackward, Softmax, CheckGrad, Train, Evaluate

import numpy as np


def InitCNN(num_channels, filter_size, num_filters_1, num_filters_2,
            num_outputs):
    """Initializes CNN parameters.

    Args:
        num_channels:  Number of input channels.
        filter_size:   Filter size.
        num_filters_1: Number of filters for the first convolutional layer.
        num_filters_2: Number of filters for the second convolutional layer.
        num_outputs:   Number of output units.

    Returns:
        model:         Randomly initialized network weights.
    """
    W1 = 0.1 * np.random.randn(filter_size, filter_size,
                               num_channels,  num_filters_1)
    W2 = 0.1 * np.random.randn(filter_size, filter_size,
                               num_filters_1, num_filters_2)
    W3 = 0.01 * np.random.randn(num_filters_2 * 64, num_outputs)
    b1 = np.zeros((num_filters_1))
    b2 = np.zeros((num_filters_2))
    b3 = np.zeros((num_outputs))
    model = {
        'W1': W1,
        'W2': W2,
        'W3': W3,
        'b1': b1,
        'b2': b2,
        'b3': b3,
        'v_dW1': np.zeros((filter_size, filter_size, num_channels,  num_filters_1)),
        'v_dW2': np.zeros((filter_size, filter_size, num_filters_1, num_filters_2)),
        'v_dW3': np.zeros((num_filters_2 * 64, num_outputs)),
        'v_db1': np.zeros(num_filters_1),
        'v_db2': np.zeros(num_filters_2),
        'v_db3': np.zeros(num_outputs)
    }

    return model


def MaxPool(x, ratio):
    """Computes non-overlapping max-pooling layer.

    Args:
        x:     Input values.
        ratio: Pooling ratio.

    Returns:
        y:     Output values.
    """
    xs = x.shape
    h = x.reshape([xs[0], int(xs[1] / ratio), ratio,
                   int(xs[2] / ratio), ratio, xs[3]])
    y = np.max(np.max(h, axis=4), axis=2)
    return y


def MaxPoolBackward(grad_y, x, y, ratio):
    """Computes gradients of the max-pooling layer.

    Args:
        grad_y: Gradients wrt. the inputs.
        x:      Input values.
        y:      Output values.

    Returns:
        grad_x: Gradients wrt. the inputs.
    """
    dy = grad_y
    xs = x.shape
    ys = y.shape
    h = x.reshape([xs[0], int(xs[1] / ratio), ratio,
                   int(xs[2] / ratio), ratio, xs[3]])
    y_ = np.expand_dims(np.expand_dims(y, 2), 4)
    dy_ = np.expand_dims(np.expand_dims(dy, 2), 4)
    dy_ = np.tile(dy_, [1, 1, ratio, 1, ratio, 1])
    dx = dy_ * (y_ == h).astype('float')
    dx = dx.reshape([ys[0], ys[1] * ratio, ys[2] * ratio, ys[3]])
    return dx


def Conv2DBackward(grad_y, x, y, w):
    """Computes gradients of the convolutional layer.

    Args:
        grad_y: Gradients wrt. the inputs.
        x:      Input values.
        y:      Output values.

    Returns:
        grad_x: Gradients wrt. the inputs.
        grad_w: Gradients wrt. the weights.
    """
    ###########################
    # Insert your code here.
    I = w.shape[0]
    J = w.shape[1]
    w_transpose = np.transpose(w, [0, 1, 3, 2]) #[I, J, K, C]
    w_transpose = np.flipud(w_transpose) #[I, J-j+1, K, C]
    w_transpose = np.fliplr(w_transpose) #[I-i+1, J-j+1, K, C]



    grad_x = Conv2D(grad_y, w_transpose, [I-1, J-1])
    # grad_x = Conv2D(grad_y, w_transpose, [4, 4])
    grad_y_transpose = np.transpose(grad_y, [1, 2, 0, 3])
    x_transpose = np.transpose(x, [3, 1, 2, 0])

    grad_w = Conv2D(x_transpose, grad_y_transpose, [4, 4])
    # grad_w = Conv2D(grad_x_transpose, grad_y_transpose, [I-1, J-1])
    grad_w = np.transpose(grad_w, [1, 2, 0, 3])

    return grad_x, grad_w
    ###########################
    # raise Exception('Not implemented')


def CNNForward(model, x):
    """Runs the forward pass.

    Args:
        model: Dictionary of all the weights.
        x:     Input to the network.

    Returns:
        var:   Dictionary of all intermediate variables.
    """
    x = x.reshape([-1, 48, 48, 1])
    h1c = Conv2D(x, model['W1']) + model['b1']
    h1r = ReLU(h1c)
    h1p = MaxPool(h1r, 3)
    h2c = Conv2D(h1p, model['W2']) + model['b2']
    h2r = ReLU(h2c)
    h2p = MaxPool(h2r, 2)
    h2p_ = np.reshape(h2p, [x.shape[0], -1])
    y = Affine(h2p_, model['W3'], model['b3'])
    var = {
        'x': x,
        'h1c': h1c,
        'h1r': h1r,
        'h1p': h1p,
        'h2c': h2c,
        'h2r': h2r,
        'h2p': h2p,
        'h2p_': h2p_,
        'y': y
    }
    return var


def CNNBackward(model, err, var):
    """Runs the backward pass.

    Args:
        model:    Dictionary of all the weights.
        err:      Gradients to the output of the network.
        var:      Intermediate variables from the forward pass.
    """
    dE_dh2p_, dE_dW3, dE_db3 = AffineBackward(err, var['h2p_'], model['W3'])
    dE_dh2p = np.reshape(dE_dh2p_, var['h2p'].shape)
    dE_dh2r = MaxPoolBackward(dE_dh2p, var['h2r'], var['h2p'], 2)
    dE_dh2c = ReLUBackward(dE_dh2r, var['h2c'], var['h2r'])
    dE_dh1p, dE_dW2 = Conv2DBackward(
        dE_dh2c, var['h1p'], var['h2c'], model['W2'])
    dE_db2 = dE_dh2c.sum(axis=2).sum(axis=1).sum(axis=0)
    dE_dh1r = MaxPoolBackward(dE_dh1p, var['h1r'], var['h1p'], 3)
    dE_dh1c = ReLUBackward(dE_dh1r, var['h1c'], var['h1r'])
    _, dE_dW1 = Conv2DBackward(dE_dh1c, var['x'], var['h1c'], model['W1'])
    dE_db1 = dE_dh1c.sum(axis=2).sum(axis=1).sum(axis=0)
    model['dE_dW1'] = dE_dW1
    model['dE_dW2'] = dE_dW2
    model['dE_dW3'] = dE_dW3
    model['dE_db1'] = dE_db1
    model['dE_db2'] = dE_db2
    model['dE_db3'] = dE_db3
    pass


def CNNUpdate(model, eps, momentum):
    """Update NN weights.

    Args:
        model:    Dictionary of all the weights.
        eps:      Learning rate.
        momentum: Momentum.
    """
    ###########################
    # Insert your code here.
    # Update the weights.
    model['v_dW1'] = momentum * model['v_dW1'] + eps* model['dE_dW1']
    model['v_dW2'] = momentum * model['v_dW2'] + eps* model['dE_dW2']
    model['v_dW3'] = momentum * model['v_dW3'] + eps* model['dE_dW3']
    model['v_db1'] = momentum * model['v_db1'] + eps* model['dE_db1']
    model['v_db2'] = momentum * model['v_db2'] + eps* model['dE_db2']
    model['v_db3'] = momentum * model['v_db3'] + eps* model['dE_db3']

    # Update the weights.
    model['W1'] -= model['v_dW1']
    model['W2'] -= model['v_dW2']
    model['W3'] -= model['v_dW3']
    model['b1'] -= model['v_db1']
    model['b2'] -= model['v_db2']
    model['b3'] -= model['v_db3']
    ###########################
    # raise Exception('Not implemented')


def main():
    """Trains a CNN."""
    model_fname = 'cnn_model.npz'
    stats_fname = 'cnn_stats.npz'

    # Hyper-parameters. Modify them if needed.
    eps = 0.1
    momentum = 0.9
    num_epochs = 30
    filter_size = 5
    num_filters_1 = 32
    num_filters_2 = 16
    batch_size = 100

    # Input-output dimensions.
    num_channels = 1
    num_outputs = 7


    # Initialize model.
    model = InitCNN(num_channels, filter_size, num_filters_1, num_filters_2,
                    num_outputs)

    # Uncomment to reload trained model here.
    # model = Load(model_fname)

    # Check gradient implementation.
    print('Checking gradients...')
    x = np.random.rand(10, 48, 48, 1) * 0.1
    CheckGrad(model, CNNForward, CNNBackward, 'W3', x)
    CheckGrad(model, CNNForward, CNNBackward, 'b3', x)
    CheckGrad(model, CNNForward, CNNBackward, 'W2', x)
    CheckGrad(model, CNNForward, CNNBackward, 'b2', x)
    CheckGrad(model, CNNForward, CNNBackward, 'W1', x)
    CheckGrad(model, CNNForward, CNNBackward, 'b1', x)

    # Train model.
    stats = Train(model, CNNForward, CNNBackward, CNNUpdate, eps,
                  momentum, num_epochs, batch_size)

    # Uncomment if you wish to save the model.
    # Save(model_fname, model)

    # Uncomment if you wish to save the training statistics.
    # Save(stats_fname, stats)

if __name__ == '__main__':
    main()
    input("Pause for Figures")
