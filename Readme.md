Custom RandomSearchCV Implementation:
This repository contains a custom implementation of the Randomized Search Cross-Validation (RandomSearchCV) algorithm for hyperparameter tuning, specifically focusing on the hyperparameter K for the K-Nearest Neighbors (KNN) classifier. The implemented RandomSearchCV algorithm helps find an optimal K value for the KNN classifier by sampling odd numbers within a specified range.

Key Functions:
sample_odd_nums(param_range)
Returns a single odd K value within the specified hyperparameter range.
Utilizes random sampling and recursion to ensure an odd value is chosen.
sample_K_values(param_range, num_of_K)
Samples multiple odd K values using the sample_odd_nums function.
Returns a list of unique odd K values.
subtract_sample(lst_1, lst_2, sample_len)
Removes duplicates from lst_1 using elements from lst_2.
Samples sample_len points from the resulting list.
sample_indices(x_train, folds)
Samples indices for splitting the dataset into folds for cross-validation.
Returns a list of sampled indices for each fold and a complete set of indices.
RandomSearchCV(x_train, y_train, classifier, param_range, folds)
Implements the RandomSearchCV algorithm for KNN hyperparameter tuning.
Samples K values using sample_K_values and indices using sample_indices.
Iterates through folds, trains KNN with different K values, and calculates accuracy scores.
Returns mean accuracy scores for both train and test evaluations, along with the list of K values.

Usage:
Import necessary libraries and define your dataset:
numpy
random
make_classification
train_test_split
KNeighborsClassifier
accuracy_score
matplotlib.pyplot
warnings

Define the dataset and hyperparameters:
Create a dataset using make_classification.
Split the dataset into training and testing sets.

Instantiate the KNeighborsClassifier:
Create a KNN classifier object.
Define the hyperparameter range and number of folds.

Apply the RandomSearchCV function:
Call the RandomSearchCV function with appropriate arguments.

Visualize the results:
Plot the hyperparameter values against mean train and test accuracy scores using matplotlib.

Plot the decision boundary:
Define a function plot_decision_boundary to visualize the decision boundary of the trained classifier.

Example:
Refer to the provided code for a complete implementation and example usage of the custom RandomSearchCV algorithm for hyperparameter tuning of the KNN classifier. The example dataset and visualization showcase the algorithm's effectiveness in finding an optimal K value.