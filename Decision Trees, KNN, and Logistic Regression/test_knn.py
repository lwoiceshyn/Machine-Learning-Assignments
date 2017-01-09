# Leo Woiceshyn, Student Number 998082159, for CSC2515 Assignment 1

import numpy as np
import utils as ut
import matplotlib.pyplot as plt
from run_knn import run_knn

#Load data
train_data, train_labels = ut.load_train()
valid_data, valid_labels = ut.load_valid()
test_data, test_labels = ut.load_test()

#Create empty arrays for accuracy values
validation_accuracies = []
test_accuracies = []

#List for k
k_values = [1,3,5,7,9]

#Validation Set
for k in k_values:

    correct_predictions = 0
    total_predictions = 0
    predicted_valid_labels = run_knn(k, train_data, train_labels, valid_data)

    #Iterate through the predicted labels and compare them to the true labels to determine validation accuracy
    for index, value in enumerate(predicted_valid_labels):
        if predicted_valid_labels[index] == valid_labels[index]:
            correct_predictions += 1
            total_predictions += 1
        else:
            total_predictions += 1
    validation_accuracies.append(100*(float(correct_predictions) / total_predictions))

#Test Set
for k in k_values:

    correct_predictions = 0
    total_predictions = 0
    predicted_test_labels = run_knn(k, train_data, train_labels, test_data)



    #Iterate through the predicted labels and compare them to the true labels to determine test accuracy
    for index, value in enumerate(predicted_test_labels):
        # print index
        # print value
        # print test_labels[index]
        if predicted_test_labels[index] == test_labels[index]:
            correct_predictions += 1
            total_predictions += 1
        else:
            total_predictions += 1
    test_accuracies.append(100*(float(correct_predictions) / total_predictions))

print "Validation Accuracies:", validation_accuracies
print "Test Accuracies:", test_accuracies

validation_accuracies = np.array(validation_accuracies)
test_accuracies = np.array(test_accuracies)
validation_k = np.array(k_values)
test_k = np.array(k_values)

#Validation
plt.figure(1)
plt.bar(validation_k, validation_accuracies, lw=2, label = "Validation")
plt.axis([0, 11, 90, 100])
x_ticks = [1.4, 3.4, 5.4, 7.4, 9.4]
plt.xticks(x_ticks, k_values)
plt.legend(loc=1)
#Test
plt.figure(2)
plt.bar(test_k, test_accuracies, lw=2, color="green", label = "Test")
plt.axis([0, 11, 90, 100])
plt.xticks(x_ticks, k_values)
plt.xlabel('k Nearest Neighbours')
plt.ylabel('Accuracy(%)')
plt.title('Accuracy vs k')
plt.legend(loc=1)

plt.show()
