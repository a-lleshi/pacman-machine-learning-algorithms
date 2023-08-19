# classifier.py
# Lin Li/26-dec-2021
#
# Use the skeleton below for the classifier and insert your code here.

import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split

class NaiveBayes():
    def fit(self, data, target):
        self.classes = np.unique(target)  # Store the unique target classes
        self.data = data  # Store the training data
        self.target = target  # Store the target data
        self.class_counts = defaultdict(int)  # Store the class counts
        self.class_feature_counts = defaultdict(lambda: defaultdict(int))  # Store the class-feature counts
        self.class_probabilities = defaultdict(float)  # Store the class probabilities
        self.feature_probabilities = defaultdict(lambda: defaultdict(float))  # Store the feature probabilities
        
        # Calculate the class and class-feature counts
        for i in range(len(data)):
            c = target[i]
            self.class_counts[c] += 1
            for j in range(len(data[i])):
                self.class_feature_counts[c][j] += data[i][j]
        
        # Calculate the class and feature probabilities
        for c in self.classes:
            self.class_probabilities[c] = self.class_counts[c] / len(target)
            for j in range(len(data[0])):
                self.feature_probabilities[c][j] = (self.class_feature_counts[c][j] + 1) / (self.class_counts[c] + 2)
        
    def predict(self, data, legal=None):
        predictions = []  # Store the predictions
        
        # Loop through each data sample
        for d in data:
            class_scores = defaultdict(float)  # Store the class scores
            
            # Loop through each class
            for c in self.classes:
                class_scores[c] = np.log(self.class_probabilities[c])
                for j in range(len(d)):
                    if d[j] == 1:
                        class_scores[c] += np.log(self.feature_probabilities[c][j])
                    else:
                        class_scores[c] += np.log(1 - self.feature_probabilities[c][j])
                        
            # Predict the class with the highest score
            prediction = max(class_scores, key=class_scores.get)
            predictions.append(prediction)
        
        return predictions
        
        

def euclidean_distance(x1, x2):
    # Convert the input vectors to NumPy arrays
    x1 = np.array(x1)
    x2 = np.array(x2)
    # Calculate the Euclidean distance between the two input vectors using NumPy
    return np.sqrt(np.sum((x1 - x2)**2))


class KNNClassifier():
    def __init__(self):
        # Set the default number of neighbors to consider
        self.k = 5

    def reset(self):
        # Reset the number of neighbors to the default
        self.k = 5
    
    def fit(self, data, target):
        # Store the training data and target values
        self.X_train = data
        self.y_train = target

    def predict(self, data, legal=None):
        # Initialize an empty list to store the predicted target values
        y_pred = []
        # Initialize an empty list to store the distances between the input data and the training data
        distances = []

        for x in np.array(data).reshape(1,-1):
            for x_train in self.X_train:
                # Calculate the Euclidean distance between the input data and each training data point
                distances.append(euclidean_distance(x, x_train))
            
            # Find the indices of the k training data points with the smallest distances to the input data
            k_index = np.argsort(distances)[:self.k]
            # Get the target values of the k nearest neighbors
            k_neighbors = [self.y_train[i] for i in k_index]
            # Add the most common target value among the k nearest neighbors to the list of predicted target values
            y_pred.append(np.bincount(k_neighbors).argmax())
        return y_pred
  
class Node:
    ''' A node in a decision tree. '''
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):

        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTreeClassifier():
    ''' A decision tree classifier. '''
    def __init__(self):
        self.max_depth = 2
        
        # self.classes = ['North', 'East', 'South', 'West']

    def gini_impurity(self, y):
        ''' Calculate the Gini impurity of an array. '''
        classes = set(y)
        gini_impurity = 1.0
        for class_label in classes:
            probability_of_class = (y == class_label).mean()
            gini_impurity -= probability_of_class ** 2
        return gini_impurity

    def impurity(self, y_left, y_right, n_samples):
        ''' Calculate the impurity of a split. '''
        return (len(y_left) / n_samples) * self.gini_impurity(y_left) + (len(y_right) / n_samples) * self.gini_impurity(y_right)

    def build_tree(self, X, y, max_depth):
        ''' Build a decision tree by recursively finding the best split. '''

        # Convert data to numpy arrays
        X = np.array(X)
        y = np.array(y)

        # Get the number of samples and features and the labels
        n_samples, n_features = X.shape
        n_labels = len(set(y))

        # Terminate node if maximum depth is reached or if there are no more samples to split
        if n_samples <= 1 or max_depth == 0 or n_labels == 1:
            return Node(value=y)

        # Finding best split for the array of features and threshold to split on
        best_feature, best_threshold, best_impurity = 1000, 1000, 1.0
        for feature in range(n_features):
            thresholds = set(X[:, feature])
            for threshold in thresholds:
                y_left = y[X[:, feature] <= threshold]
                y_right = y[X[:, feature] > threshold]

                impurity = self.impurity(y_left, y_right, n_samples)
                if impurity < best_impurity:
                    best_feature, best_threshold, best_impurity = feature, threshold, impurity

        # Build the left and right subtrees recursively taking 1 off the maximum depth
        X_left, y_left = X[X[:, best_feature] <= best_threshold], y[X[:, best_feature] <= best_threshold]
        X_right, y_right = X[X[:, best_feature] > best_threshold], y[X[:, best_feature] > best_threshold]

        left = self.build_tree(X_left, y_left, max_depth - 1)
        right = self.build_tree(X_right, y_right, max_depth - 1)

        return Node(feature=best_feature, threshold=best_threshold, left=left, right=right)

    # prediction of one data type for pacman to move into one direction from the node it self 
    def predict(self, node, x):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self.predict(node.left, x)
        else:
            return self.predict(node.right, x)

# Create a VotingClassifier class
class VotingClassifier:
    
    # Constructor that accepts a list of estimator objects
    def __init__(self, estimators):
        self.estimators = estimators
        
    # Method to fit the model with the provided training data and target labels
    def fit(self, X, y):
        # For each estimator in the list, call its 'fit' method with the provided data
        for estimator in self.estimators:
            estimator.fit(X, y)
    
    # Method to predict the target labels for the provided data
    def predict(self, X):
        # For each estimator in the list, call its 'predict' method and store the predictions in a list
        preds = [estimator.predict(X) for estimator in self.estimators]
        
        # Find the most common prediction for each sample by taking the mode of the predictions
        # for each estimator, and return the list of predicted labels

        return preds



#SVM Classifier has not been used in the final prediction, because 
#it predicts "1" for every prediction.

class SVMClassifier:
    def __init__(self, C=1.0, kernel='poly', gamma=1.0):
        self.C = C 
        self.kernel = kernel
        self.gamma = gamma
        self.classifiers = []
        pass
    
    def fit(self, data, target):
        self.data = data  # Store the training data
        self.target = target  # Store the target data
        classifiers = np.unique(target)
        directions = len(classifiers)
        # Iterate through each pair of classifiers, will be 6 iterations due to 4 chose 2 
        for i in range (directions):
            for j in range(i+1, directions):
                data = np.array(data)
                target = np.array(target)
                # Select only the samples in the data that belong to one of the two classes being considered by the current binary classifier
                grab_Data = np.logical_or(np.any(target == classifiers[i]), np.any(target == classifiers[j]))
                # Select the rows of data that correspond to the samples in the current binary classification 
                X = data[grab_Data, :]
                 # Select the rows of target values that correspond to the samples in the current binary classification 
                y = target[grab_Data]
                 # For an svm to work we need to make them each a binary case for the classifier, 
                 # Each pair of classifiers have to be distinguishable thus give one a value of -1 and the other 1
                y[y == classifiers[i]] = -1
                y[y== classifiers[j]] = 1
                # Since we split up the pairs we need to now pass the data through an svm classifier and the iteration will allow us to do all the pairs 
                supportVectorMachine = binaryClassifier(C=self.C, kernel=self.kernel, gamma=self.gamma)
                supportVectorMachine.fit(X, y)
                self.classifiers.append((supportVectorMachine, classifiers[i], classifiers[j]))

    def predict(self, data, legal = None):
        if legal is not None:
            legal = np.reshape(legal, (1, len(legal)))
        # Get the number of input samples and binary classifiers
        data = np.array(data)
        
        if len(data.shape) == 2:
            number_of_samples = data.shape[0]
        number_of_classifiers = len(self.classifiers)

        # Create an empty array to store the predicted class scores
        scores = np.zeros((number_of_samples, number_of_classifiers))

        # Loop over all binary classifiers and compute the decision function score for each input sample
        for i in range(number_of_classifiers):
            # Get the binary classifier and the two classes it distinguishes between
            classifier, class1, class2 = self.classifiers[i]

            # Compute the decision function score for each input using the binary classifier
            score = classifier.decision_function(data)
        
            # Assign the predicted class based on the decision function score
            predicted_classes = np.where(score >= 0, class2, class1)

            # Store the predicted classes for the current binary classifier
            scores[:, i] = predicted_classes

        # Compute the final predicted class for each input sample based on the predicted classes for each binary classifier
        predictions = np.zeros(number_of_samples)
       
        for i in range(number_of_samples):
            
            predicted_classes = scores[i, :]
           
            counts = np.bincount(predicted_classes.astype('int'))
           
            predicted_class = np.argmax(counts)
           
         
        # Return the final predicted class for each input sample
        return [predicted_class]


# We use gradient descent to compare the 2 classes
class binaryClassifier:
    def __init__(self, C=1.0, kernel='linear', gamma=0.1, learning_rate=0.01, numberOfIterations=100, tradeOff_parameter=0.1):
        self.learning_rate = learning_rate
        self.numberOfIterations = numberOfIterations
        self.tradeOff_parameter = tradeOff_parameter

    def fit(self, data, target):

        data = data[0]
        target = target[0]
        self.weights = np.zeros(data.shape[1])
        self.bias = 0

        # Gradient descent
        for i in range(self.numberOfIterations):
            # Calculate predicted values
            y_pred = self.predict(data, legal=None)

            # Calculate gradient of cost function
           
            weight = np.dot(data.T, y_pred - target) / data.shape[0] + self.tradeOff_parameter * self.weights / data.shape[0]
            
            bias = np.sum(y_pred - target) / data.shape[0]

            # Update weight and bias
            self.weights -= self.learning_rate * weight
            self.bias -= self.learning_rate * bias

    def predict(self, data, legal=None):
        if legal is not None:
            data = np.array(data)
            data = np.concatenate((data, legal), axis=0)

        # Calculate the dot product of the input data and the weights
        binary_model = np.dot(data, self.weights) + self.bias

        # Apply the sign function to get the predicted classes
        # Compute probability score 
        y_pred = self.softmax(binary_model) 
        y = np.argmax(y_pred)
        
        return y

    # Generatingg unnormalised scores 
    def decision_function(self, X):
        return np.dot(X, self.weights) + self.bias
            
    # Normalises the scores generated by the decision function
    def softmax(self, x):
        return np.exp(x)/ np.exp(x).sum()      

class Classifier():
    def __init__(self):
        #Initialise the DT classifier
        self.model = DecisionTreeClassifier()
        self.model_fit = None

        #Initialise classifiers suitable for the Voting Classifier
        self.knn = KNNClassifier()
        self.nb = NaiveBayes()  # Initialize the NaiveBayes classifier
        
        #SVM is flawed, is not used
        # self.svm = SVMClassifier()
        
        #Naive Bayes and KNN classifiers used in the ensemble, 
        #SVM is excluded because returns "1" for every move. 
        self.estimators = [self.nb, self.knn]
        self.ensemble = VotingClassifier(self.estimators)
        

    def reset(self):
        #Reset all the classifiers
        self.model = DecisionTreeClassifier()
        self.model_fit = None
        
        self.knn = KNNClassifier()
        self.nb = NaiveBayes()  # Initialize the NaiveBayes classifier

        self.estimators = [self.nb, self.knn]
        self.ensemble = VotingClassifier(self.estimators)

    
    def fit(self, data, target):    
        # split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)
        self.model_fit = self.model.build_tree(X_train, y_train, self.model.max_depth)

        #Fir the ensemble of classifiers SVM, KNN and NB
        self.ensemble.fit(data, target)

    # Predict based on argmax
    def predict(self, data, legal=None):
        #Decision Tree Classifier prediction
        predictions = self.model.predict(self.model_fit, data)
        class_counts = np.bincount(predictions)
        most_frequent_class = np.argmax(class_counts)


        #Decision Tree Classifier predicted most frequent class appended 
        # to Voting Classifiers Prediction
        ensemble = self.ensemble.predict(np.array(data).reshape(1, -1))
        combined = np.append(ensemble,most_frequent_class)
      
        print("Combined Predictions: ", combined)
        argmax_of_all = np.bincount(combined).argmax()
        
        print("Final Prediction: ", argmax_of_all)
        return argmax_of_all



