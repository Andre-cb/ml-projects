"""
  Authors: Andre Costa Barros, Seamus Delaney
  Description:
  This implementation of AdaBoost combines multiple weighted weak linear learners to create a strong classifier.
  The algorithm iteratively trains weak learners on weighted data, adjusting weights based on misclassifications.
  Weak learners are combined into a strong classifier by weighted voting.
  The process includes tracking accuracy and visualizing decision boundaries, all without external ML libraries.
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class weakLearner:
    """
    A weak learner for AdaBoost algorithm. It implements a simple linear classifier
    using weighted means to create an orientation vector, project data points,
    and find the best decision boundary.
    """
    def __init__(self, dataset):
        """
        Initializes the weak learner with the provided dataset.

        Parameters:
            dataset (pd.DataFrame): Dataset containing features and labels.
        """
        self.dataset = dataset
        self.decisionBoundry = None  # To be set during training
        self.amount_of_say = None  # To be set during training
        self.normalisedOrientationVector = None  # To be set during training

    def fit(self):
        """
        Fits the weak learner by calculating weighted class means, orientation vector,
        and projections of data points.
        """
        # Initialize means as numpy arrays for element-wise operations
        weightedNegativeClassMean = np.zeros(2)
        weightedPositiveClassMean = np.zeros(2)

        # Calculate weighted means for each class
        for i in range(len(self.dataset)):
            if self.dataset['output'].iloc[i] == 1:  # Positive class
                weightedPositiveClassMean += self.dataset[['feature1', 'feature2']].iloc[i].values * self.dataset['weight'].iloc[i]
            elif self.dataset['output'].iloc[i] == -1:  # Negative class
                weightedNegativeClassMean += self.dataset[['feature1', 'feature2']].iloc[i].values * self.dataset['weight'].iloc[i]

        # Calculate the orientation vector (difference between means)
        orientationVector = weightedPositiveClassMean - weightedNegativeClassMean

        # Compute ||r|| (magnitude of the orientation vector)
        orientationVectorMagnitude = np.linalg.norm(orientationVector)

        # Normalize the orientation vector
        normalisedOrientationVector = orientationVector / orientationVectorMagnitude
        self.normalisedOrientationVector = normalisedOrientationVector

        # Projections of the points onto the normalized orientation vector
        projections = self.dataset['feature1'].values * normalisedOrientationVector[0] + self.dataset['feature2'].values * normalisedOrientationVector[1]

        # Add projections directly to the dataset
        self.dataset['projections'] = projections.astype(float)

        # Sort the dataset by the projections column
        self.dataset.sort_values(by='projections', inplace=True)

        # Output results for debugging
        print("Weighted Negative Class Mean:", weightedNegativeClassMean)
        print("Weighted Positive Class Mean:", weightedPositiveClassMean)
        print("Orientation Vector:", orientationVector)

    def find_best_midpoint(self):
        """
        Finds the best decision boundary (midpoint) by calculating weighted misclassification error
        and selecting the best boundary.

        Returns:
            bestMidpoint (float): The best decision boundary.
            amount_of_say (float): The amount of influence the weak learner has on the final prediction.
            misclassifiedPoints (list): Indices of the misclassified points.
        """
        lowestError = float('inf')
        bestMidpoint = None
        misclassifiedPoints = []

        # Extract sorted projections, labels (output), and weights (weight)
        sorted_projections = self.dataset['projections'].values
        sorted_outputs = self.dataset['output'].values
        sorted_weights = self.dataset['weight'].values

        # Iterate over possible midpoints
        for i in range(len(sorted_projections) - 1):
            midpoint = (sorted_projections[i] + sorted_projections[i + 1]) / 2

            # Calculate weighted misclassification error for this midpoint
            misclassificationWeightSum = 0
            totalWeightSum = 0
            currentMisclassifiedPoints = []  # Track misclassified points for this midpoint

            for j in range(len(sorted_projections)):
                if sorted_projections[j] > midpoint:
                    if sorted_outputs[j] == -1:  # Misclassified negative class
                        misclassificationWeightSum += sorted_weights[j]
                        currentMisclassifiedPoints.append(j)
                else:
                    if sorted_outputs[j] == 1:  # Misclassified positive class
                        misclassificationWeightSum += sorted_weights[j]
                        currentMisclassifiedPoints.append(j)
                totalWeightSum += sorted_weights[j]

            error = misclassificationWeightSum / totalWeightSum

            # Check if this is the best midpoint (lowest error)
            if error < lowestError:
                lowestError = error
                bestMidpoint = midpoint
                misclassifiedPoints = currentMisclassifiedPoints

            amount_of_say = 0.5 * np.log((1 - lowestError) / lowestError)

        # Store the best midpoint and amount of say in the weak learner
        self.decisionBoundry = bestMidpoint
        self.amount_of_say = amount_of_say

        return bestMidpoint, amount_of_say, misclassifiedPoints

def initalise_weights(dataset):
    """
    Initializes weights for the dataset based on the class distribution.
    Each class gets a weight inversely proportional to the class size.

    Parameters:
        dataset (pd.DataFrame): The dataset with features and labels.
    """
    num_negatives = (dataset['output'] == -1).sum()
    num_positives = (dataset['output'] == 1).sum()

    # Initialize weights
    weights = []
    for i in range(len(dataset)):
        if dataset['output'].iloc[i] == -1:  # Negative class
            weights.append(1 / num_negatives)
        else:  # Positive class
            weights.append(1 / num_positives)
    dataset['weight'] = weights

def find_number_of_weak_learners(trainset):
    """
    Trains weak learners iteratively and adds them until 100% accuracy is achieved on the training set.

    Parameters:
        trainset (pd.DataFrame): The training dataset with features, labels, and weights.

    Returns:
        tuple: (best_predictions, weak_learners, best_accuracy)
    """
    # Initialize variables for tracking
    weak_learners = []
    best_predictions = None
    best_accuracy = 0  # Start with 0 accuracy

    # Initialize weights for the training set
    initalise_weights(trainset)
    i = 0

    while best_accuracy < 100:  # Keep training until 100% accuracy
        # Print iteration
        i += 1
        print(f"Iteration: {i}")

        # Train a weak learner on the full training set
        weak_learner = weakLearner(trainset)
        weak_learner.fit()
        bestMidpoint, amount_of_say, misclassifiedPoints = weak_learner.find_best_midpoint()

        # Add the weak learner to the list
        weak_learners.append(weak_learner)

        # Calculate predictions and accuracy on the training set after adding the weak learner
        predictions, accuracy = classify_samples(trainset, weak_learners)

        # Update the best accuracy
        best_accuracy = accuracy
        best_predictions = predictions  # Update best predictions

        # Update weights in the training set using misclassified points
        get_new_weights(trainset, misclassifiedPoints, amount_of_say)

        # Print Accuracy of current Iteration
        print(f"Accuracy: {best_accuracy:.2f}%\n")

    return best_predictions, weak_learners, best_accuracy

def get_new_weights(dataset, misclassified, amount_of_say):
    """
    Updates the 'weight' column of the dataset based on misclassified points and the weak learner's influence.

    Parameters:
        dataset (pd.DataFrame): The original dataset with weights to update.
        misclassified (list): Indices of misclassified points.
        amount_of_say (float): The weak learner's weight.
    """
    for i in range(len(dataset)):
        if i in misclassified:
            dataset.iloc[i, dataset.columns.get_loc('weight')] *= math.exp(amount_of_say)
        else:
            dataset.iloc[i, dataset.columns.get_loc('weight')] *= math.exp(-amount_of_say)

    # Normalize the weights so they sum to 1
    total_weight = dataset['weight'].sum()
    dataset['weight'] /= total_weight

def classify_samples(trainset, weak_learners):
    """
    Classifies samples using the trained weak learners and calculates accuracy.

    Parameters:
        dataset (pd.DataFrame): The dataset to classify (e.g., testset), including true labels.
        weak_learners (list): List of trained weak learners.

    Returns:
        tuple: (final_predictions, accuracy)
            - final_predictions (np.ndarray): The predicted class labels (+1 or -1) for the dataset.
            - accuracy (float): The classification accuracy as a percentage.
    """
    feature1 = trainset['feature1'].values
    feature2 = trainset['feature2'].values
    expected_output = trainset['output'].values

    # Initialize combined scores for all samples
    combined_scores = np.zeros(len(trainset))

    # Aggregate predictions from all weak learners
    for learner in weak_learners:
        normalisedOrientationVector = learner.normalisedOrientationVector
        decisionBoundry = learner.decisionBoundry
        amount_of_say = learner.amount_of_say

        # Compute projections
        projections = feature1 * normalisedOrientationVector[0] + feature2 * normalisedOrientationVector[1]

        # Make predictions based on the decision boundary
        predictions = np.where(projections > decisionBoundry, 1, -1)

        # Weight predictions by the weak learner's amount of say
        combined_scores += amount_of_say * predictions

    # Final predictions based on the sign of the combined scores
    final_predictions = np.where(combined_scores > 0, 1, -1)

    # Calculate accuracy
    correct_predictions = np.sum(final_predictions == expected_output)
    accuracy = (correct_predictions / len(expected_output)) * 100  # Percentage

    return final_predictions, accuracy

def plot_overall_boundary(dataset, weak_learners, predictions):
    """
    Plots the overall decision boundary and binary classification regions (red for positive, blue for negative).

    Parameters:
        dataset (pd.DataFrame): Dataset containing features and labels.
        weak_learners (list): List of trained weak learners.
    """
    # Extract feature ranges for the grid
    x_min, x_max = dataset['feature1'].min() - 1, dataset['feature1'].max() + 1
    y_min, y_max = dataset['feature2'].min() - 1, dataset['feature2'].max() + 1

    # Create a grid of points
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid_points = np.c_[xx.ravel(), yy.ravel()]  # Flatten the grid into a 2D array

    # Initialize combined scores for all grid points
    combined_scores = np.zeros(len(grid_points))

    # Aggregate the contributions from all weak learners
    for learner in weak_learners:
        normalisedOrientationVector = learner.normalisedOrientationVector
        decisionBoundry = learner.decisionBoundry
        amount_of_say = learner.amount_of_say

        # Compute projections for grid points
        projections = grid_points[:, 0] * normalisedOrientationVector[0] + grid_points[:, 1] * normalisedOrientationVector[1]

        # Make predictions based on the decision boundary
        predictions = np.where(projections > decisionBoundry, 1, -1)

        # Add the weighted predictions to the combined scores
        combined_scores += amount_of_say * predictions

    # Reshape combined scores into the grid shape
    combined_scores = combined_scores.reshape(xx.shape)

    # Plot the decision regions (binary colors: blue for negative, red for positive)
    plt.contourf(xx, yy, combined_scores > 0, levels=1, colors=['blue', 'red'], alpha=0.2)

    # Plot the dataset
    plt.scatter(dataset['feature1'][dataset['output'] == -1],
                dataset['feature2'][dataset['output'] == -1], color='blue', label='Negative Class')
    plt.scatter(dataset['feature1'][dataset['output'] == 1],
                dataset['feature2'][dataset['output'] == 1], color='red', label='Positive Class')

    # Plot the overall decision boundary (solid line where combined_scores = 0)
    plt.contour(xx, yy, combined_scores, levels=[0], colors='black', linewidths=2)

    # Add labels, legend, and title
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Overall Decision Boundary with Binary Classification Regions')
    plt.legend()
    plt.grid()
    plt.show()

def main():
    # Load the training set and test set
    trainset = pd.read_csv('adaboost-train-24.txt', sep='\s+', header=None, names=['feature1', 'feature2', 'output'])
    testset = pd.read_csv('adaboost-test-24.txt', sep='\s+', header=None, names=['feature1', 'feature2', 'output'])

    # Initialize weights for the training set
    initalise_weights(trainset)

    # Train weak learners and find the optimal number for 100% accuracy
    train_best_predictions, weak_learners, train_best_accuracy = find_number_of_weak_learners(trainset)

    # Plot the overall decision boundary for the training set
    plot_overall_boundary(trainset, weak_learners, train_best_predictions)

    # Test the weak learners on the test set
    test_predictions, test_accuracy = classify_samples(testset, weak_learners)
    plot_overall_boundary(testset, weak_learners, test_predictions)

    # Print results
    print(f"Train Accuracy: {train_best_accuracy:.2f}%")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Number of weak learners: {len(weak_learners)}")

if __name__ == "__main__":
    main()