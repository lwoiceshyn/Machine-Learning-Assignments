# Leo Woiceshyn, Student Number 998082159, for CSC2515 Assignment 1
from check_grad import check_grad
from utils import *
from logistic import *
import matplotlib.pyplot as plt

global iterations , valid_or_test
iterations = 0


def run_logistic_regression(hyperparameters):
    global iterations, valid_or_test


    #Comment out one of these based on small or large training set
    train_inputs, train_targets = load_train()
    # train_inputs, train_targets = load_train_small()

    #Comment out one set of these based on validation or test set
    valid_inputs, valid_targets = load_valid()
    valid_or_test = 0

    # valid_inputs, valid_targets = load_test()
    # valid_or_test = 1

    # N is number of examples; M is the number of features per example.
    N, M = train_inputs.shape




    # Logistic regression weights
    # TODO:Initialize to random weights here.
    weights = np.zeros((M+1,1))
    #weights = np.random.randn(M+1, 1)
    #weights = np.random.randint(0,10, (M+1,1))

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)

    # Begin learning with gradient descent
    logging = np.zeros((hyperparameters['num_iterations'], 5))
    for t in xrange(hyperparameters['num_iterations']):


        # Find the negative log likelihood and its derivatives w.r.t. the weights.
        f, df, predictions = logistic(weights, train_inputs, train_targets, hyperparameters)


        # Evaluate the prediction.
        cross_entropy_train, frac_correct_train = evaluate(train_targets, predictions)

        if np.isnan(f) or np.isinf(f):
            raise ValueError("nan/inf error")

        # update parameters
        # weights_old = weights
        weights = weights - hyperparameters['learning_rate'] * df / N

        # if sum(abs(weights_old - weights)) > 0.05:
        #     pass
        # else:
        #     iterations = t
        #     break



        # Make a prediction on the valid_inputs.
        predictions_valid = logistic_predict(weights, valid_inputs)

        # Evaluate the prediction.
        cross_entropy_valid, frac_correct_valid = evaluate(valid_targets, predictions_valid)



        #print some stats
        #print t+1, f / N, cross_entropy_train, frac_correct_train, cross_entropy_valid, frac_correct_valid
        print ("ITERATION:{:4d}  TRAIN NLOGL:{:4.2f}  TRAIN CE:{:.6f} "
               "TRAIN FRAC:{:2.2f}  VALID CE:{:.6f}  VALID FRAC:{:2.2f}").format(
                   t+1, float(f / N), float(cross_entropy_train), float(frac_correct_train)*100,
                   float(cross_entropy_valid), float(frac_correct_valid)*100)


        logging[t] = [f / N, cross_entropy_train, frac_correct_train*100, cross_entropy_valid, frac_correct_valid*100]



    return logging

def run_check_grad(hyperparameters):
    """Performs gradient check on logistic function.
    """

    # This creates small random data with 7 examples and 
    # 9 dimensions and checks the gradient on that data.
    num_examples = 7
    num_dimensions = 9

    weights = np.random.randn(num_dimensions+1, 1)
    data    = np.random.randn(num_examples, num_dimensions)
    targets = (np.random.rand(num_examples, 1) > 0.5).astype(int)

    # print weights
    # print data
    # print targets

    diff = check_grad(logistic,      # function to check
                      weights,
                      0.001,         # perturbation
                      data,
                      targets,
                      hyperparameters)

    print "diff =", diff

if __name__ == '__main__':
    global iterations, valid_or_test
    # TODO: Set hyperparameters
    hyperparameters = {
                    'learning_rate': 0.07,
                    'weight_regularization': True, # boolean, True for using Gaussian prior on weights
                    'num_iterations': 1000,
                    'weight_decay':0.1 # related to standard deviation of weight prior
                    }

    # average over multiple runs
    num_runs = 1
    logging = np.zeros((hyperparameters['num_iterations'], 5))
    for i in xrange(num_runs):
        logging += run_logistic_regression(hyperparameters)
    logging /= num_runs

    if iterations == 0:
        iterations = hyperparameters['num_iterations']
    iterations_plot = np.linspace(1, iterations, iterations, dtype=int)
    cev_plot = logging[:iterations,3].T
    cet_plot = logging[:iterations,1].T
    loss_plot = logging[:iterations,0].T



    if valid_or_test == 0:
        plt.figure(1)
        plt.plot(iterations_plot, cev_plot, lw=2)
        plt.xlabel('Iterations')
        plt.ylabel('Cross Entropy')
        plt.title('Validation Cross Entropy vs Iterations')
        plt.axis([0, iterations, 25, 35])


    if valid_or_test == 1:
        plt.figure(1)
        plt.plot(iterations_plot, cev_plot, lw=2)
        plt.xlabel('Iterations')
        plt.ylabel('Cross Entropy')
        plt.title('Test Cross Entropy vs Iterations')
        plt.axis([0, iterations, 0, 35])


    plt.figure(2)
    plt.plot(iterations_plot, cet_plot, lw=2, color="red")
    plt.xlabel('Iterations')
    plt.ylabel('Cross Entropy')
    plt.title('Training Cross Entropy vs Iterations')
    plt.axis([0, iterations, 0, 10])

    plt.figure(3)
    plt.plot(iterations_plot, loss_plot, lw=2, color="purple")
    plt.xlabel('Iterations')
    plt.ylabel('Loss Function')
    plt.title('Loss Function vs Iterations')
    plt.axis([0, iterations, 0, 1])
    plt.show()

