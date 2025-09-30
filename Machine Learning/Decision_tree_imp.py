import math
import numpy as np
import random
from collections import Counter
import pandas as pd

class Node:
 """
 A class to represent a single node in the decision tree.
 """
 def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
  """
  Constructor for a node.
  - For decision nodes:
   - feature_index: the index of the feature to split on
   - threshold: the threshold value for the split
   - left: the left child node
   - right: the right child node
  - For leaf nodes:
   - value: the class label for the leaf node
  """
  self.feature_index = feature_index
  self.threshold = threshold
  self.left = left
  self.right = right
  self.value = value

class DecisionTreeClassifier:
 """
 A class to implement a decision tree classifier from scratch.
 """
 def __init__(self, max_depth=None, min_samples_split=2, criterion="entropy", categorical_features=None):
  """
  Constructor for the classifier.
   - max_depth: the maximum depth of the tree to prevent overfitting.
   - min_samples_split: the minimum number of samples required to split an internal node.
   - criterion: the function to measure the quality of a split ("entropy" or "gini").
   - categorical_features: a list of indices for categorical features.
  """
  self.max_depth = max_depth
  self.min_samples_split = min_samples_split
  self.criterion = criterion
  self.categorical_features = categorical_features if categorical_features else []
  self.root = None

 def fit(self, X, y):
  """
  Builds the decision tree by recursively splitting the data.
  """
  self.root = self._grow_tree(X, y)
 
 def predict(self, X):
  """
  Makes predictions for a set of data points by traversing the tree.
  """
  return np.array([self._traverse_tree(x, self.root) for x in X])
 
 def _grow_tree(self, X, y, depth=0):
  """
  Recursive function to build the tree.
  """
  n_samples, n_features = X.shape
  n_labels = len(np.unique(y))
  
  # Stopping criteria: pure node, max depth reached, or too few samples.
  if (self.max_depth is not None and depth >= self.max_depth) or n_labels == 1 or n_samples < self.min_samples_split:
   leaf_value = self._most_common_label(y)
   return Node(value=leaf_value)

  # Find the best split
  best_split = self._find_best_split(X, y)
  if not best_split or best_split["info_gain"] <= 0:
   # If no information gain, make it a leaf node
   leaf_value = self._most_common_label(y)
   return Node(value=leaf_value)
  
  # Get the split details
  feature_idx, threshold = best_split["feature_idx"], best_split["threshold"]
  
  # Split the data
  left_idxs = best_split["left_idxs"]
  right_idxs = best_split["right_idxs"]
  
  # Recursively grow the children
  left_child = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
  right_child = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
  
  return Node(feature_index=feature_idx, threshold=threshold, left=left_child, right=right_child)
 
 def _find_best_split(self, X, y):
  """
  Finds the best split (feature and threshold) for the current data
  """
  best_gain = -1
  best_split = None
  n_features = X.shape[1]
  
  for feature_idx in range(n_features):
   feature_values = X[:, feature_idx]
  
   if feature_idx in self.categorical_features:
    # categorical → test subsets
    categories = np.unique(feature_values)
    for cat in categories:
     info_gain = self._calculate_information_gain(y, feature_values, cat, categorical=True)
     if info_gain > best_gain:
      best_gain = info_gain
      best_split = {
       "feature_idx": feature_idx,
       "threshold": cat,  # category value
       "info_gain": best_gain,
       "left_idxs": np.where(feature_values == cat)[0],
       "right_idxs": np.where(feature_values != cat)[0]
      }
   else:
    # continuous → test midpoints
    unique_vals = np.unique(feature_values)
    if len(unique_vals) == 1:
     continue
    thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2
    for threshold in thresholds:
     info_gain = self._calculate_information_gain(y, feature_values, threshold, categorical=False)
     if info_gain > best_gain:
      best_gain = info_gain
      best_split = {
       "feature_idx": feature_idx,
       "threshold": threshold,
       "info_gain": best_gain,
       "left_idxs": np.where(feature_values <= threshold)[0],
       "right_idxs": np.where(feature_values > threshold)[0]
      }
  
  return best_split
 
 def _calculate_information_gain(self, y, feature_column, threshold, categorical=False):
  """
  Calculates the information gain for a given split.
  """
  parent_impurity = self._impurity(y)
  if categorical:
   left_y = y[feature_column == threshold]
   right_y = y[feature_column != threshold]
  else:
   left_y = y[feature_column <= threshold]
   right_y = y[feature_column > threshold]
  
  if len(left_y) == 0 or len(right_y) == 0:
   return 0
  
  n = len(y)
  weighted_child_impurity = (len(left_y) / n) * self._impurity(left_y) + \
   (len(right_y) / n) * self._impurity(right_y)
  
  return parent_impurity - weighted_child_impurity
 
 def score(self, X, y):
  """
  Simple accuracy metric.
  """
  return np.mean(self.predict(X) == y)
 
 def _impurity(self, y):
  """
  Calculates the impurity (entropy or Gini) of a list of labels.
  """
  if len(y) == 0:
   return 0
  
  counts = Counter(y)
  probabilities = [count / len(y) for count in counts.values()]
  
  if self.criterion == "gini":
   return 1 - sum(p ** 2 for p in probabilities)
  else:  # entropy
   return -sum(p * math.log2(p) for p in probabilities if p > 0)
 
 def _traverse_tree(self, x, node):
  """
  Traverses the tree to find the prediction for a single data point.
  """
  if node.value is not None:
   return node.value
  
  if node.feature_index in self.categorical_features:
   if x[node.feature_index] == node.threshold:
    return self._traverse_tree(x, node.left)
   else:
    return self._traverse_tree(x, node.right)
  else:
   if x[node.feature_index] <= node.threshold:
    return self._traverse_tree(x, node.left)
   else:
    return self._traverse_tree(x, node.right)
 
 def _most_common_label(self, y):
  """
  Helper function to find the most common label in a list.
  """
  if len(y) == 0:
   return None
  counts = Counter(y)
  return counts.most_common(1)[0][0]

if __name__ == "__main__":
 # Example Usage: Create a simple dataset and test the classifier
 
 # Let's create a more complex dataset to test categorical features
    data = {
        'age': [25, 35, 45, 20, 30, 50, 60, 40],
        'is_student': ['no', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes'], # Categorical
        'income': [50000, 60000, 90000, 10000, 20000, 100000, 120000, 80000],
        'buys_computer': ['no', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes']
    }
    df = pd.DataFrame(data)
    
    # Convert 'buys_computer' to numerical label
    df['buys_computer'] = df['buys_computer'].map({'no': 0, 'yes': 1})
    
    # Define features and labels
    features = ['age', 'is_student', 'income']
    label = 'buys_computer'
    
    # Get the feature matrix X and label vector y
    X = df[features].values
    y = df[label].values
    
    # Define categorical features by their index
    categorical_features = [1] 
    
    # Simple train-test split
    split_ratio = 0.75
    split_idx = int(len(X) * split_ratio)
    
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]
    
    # Initialize and train the decision tree
    print("Training Decision Tree with max_depth=3 and Gini criterion...")
    clf = DecisionTreeClassifier(max_depth=3, min_samples_split=2, criterion="gini", categorical_features=categorical_features)
    clf.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = clf.predict(X_test)
    
    # Calculate accuracy
    accuracy = clf.score(X_test, y_test)
    
    print("\nTraining Data:")
    print("X_train:\n", X_train)
    print("y_train:", y_train)
    print("\nTest Data:")
    print("X_test:\n", X_test)
    print("y_test:", y_test)
    print("\nPredictions:", y_pred)
    print(f"\nAccuracy: {accuracy:.2f}")

class RandomForestClassifier:
 """
 A class to implement a Random Forest classifier.
 """
 def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, criterion="entropy", max_features="sqrt", categorical_features=None):
  """
  Constructor for the Random Forest.
   - n_estimators: The number of trees in the forest.
   - max_depth: The maximum depth for each tree.
   - min_samples_split: The minimum number of samples to split a node.
   - criterion: The impurity measure for each tree ("entropy" or "gini").
   - max_features: The number of features to consider for each split ("sqrt", "log2", or an integer).
   - categorical_features: List of indices for categorical features.
  """
  self.n_estimators = n_estimators
  self.max_depth = max_depth
  self.min_samples_split = min_samples_split
  self.criterion = criterion
  self.max_features = max_features
  self.categorical_features = categorical_features
  self.trees = []
  
 def fit(self, X, y):
  """
  Builds the Random Forest by training multiple decision trees.
  """
  self.trees = []
  n_samples, n_features = X.shape
  
  # Determine the number of features to use per split
  if self.max_features == "sqrt":
   self.n_features_to_use = int(np.sqrt(n_features))
  elif self.max_features == "log2":
   self.n_features_to_use = int(np.log2(n_features))
  elif isinstance(self.max_features, int):
   self.n_features_to_use = self.max_features
  else:
   self.n_features_to_use = n_features
  
  for _ in range(self.n_estimators):
   # 1. Bootstrap Aggregating (Bagging)
   # Create a bootstrapped dataset by sampling with replacement
   bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
   X_bootstrap = X[bootstrap_indices]
   y_bootstrap = y[bootstrap_indices]
   
   # 2. Feature Randomness
   # Select a random subset of features for this tree
   feature_indices = random.sample(range(n_features), self.n_features_to_use)
   
   # Create a new DecisionTreeClassifier instance
   tree = DecisionTreeClassifier(
    max_depth=self.max_depth,
    min_samples_split=self.min_samples_split,
    criterion=self.criterion,
    categorical_features=[i for i in self.categorical_features if i in feature_indices] if self.categorical_features else None
   )
   
   # Adjust the dataset to only include the selected features
   X_bootstrap_subset = X_bootstrap[:, feature_indices]
   
   # Fit the tree
   tree.fit(X_bootstrap_subset, y_bootstrap)
   self.trees.append((tree, feature_indices))
 
 def predict(self, X):
  """
  Aggregates predictions from all trees using a majority vote.
  """
  all_predictions = []
  for tree, feature_indices in self.trees:
   # Predict using only the features that the tree was trained on
   X_subset = X[:, feature_indices]
   predictions = tree.predict(X_subset)
   all_predictions.append(predictions)
  
  # Transpose the list of lists to group predictions by sample
  all_predictions = np.array(all_predictions).T
  
  # 3. Aggregate Predictions (Majority Vote)
  final_predictions = np.array([Counter(preds).most_common(1)[0][0] for preds in all_predictions])
  
  return final_predictions
 
 def score(self, X, y):
  """
  Calculates the accuracy score.
  """
  predictions = self.predict(X)
  return np.mean(predictions == y)

if __name__ == "__main__":
    data = {
        'age': [25, 35, 45, 20, 30, 50, 60, 40, 28, 55],
        'is_student': ['no', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes', 'yes', 'no'], 
        'income': [50000, 60000, 90000, 10000, 20000, 100000, 120000, 80000, 35000, 95000],
        'buys_computer': ['no', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'no', 'yes']
    }
    df = pd.DataFrame(data)
    
    # Convert 'buys_computer' to numerical label
    df['buys_computer'] = df['buys_computer'].map({'no': 0, 'yes': 1})
    
    # Define features and labels
    features = ['age', 'is_student', 'income']
    label = 'buys_computer'
    
    X = df[features].values
    y = df[label].values
    
    # Define categorical features by their index
    categorical_features = [1] 
    
    # Simple train-test split
    split_ratio = 0.8
    split_idx = int(len(X) * split_ratio)
    
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]
    
    # Initialize and train the Random Forest
    print("Training Random Forest with 10 trees...")
    rf_clf = RandomForestClassifier(
        n_estimators=10, 
        max_depth=3, 
        min_samples_split=2,
        criterion="entropy",
        max_features="sqrt", # Randomly select sqrt(3) ~= 1 feature per split
        categorical_features=categorical_features
    )
    rf_clf.fit(X_train, y_train)
    
    # Make predictions and evaluate
    y_pred = rf_clf.predict(X_test)
    accuracy = rf_clf.score(X_test, y_test)
    
    print("\nTest Data:")
    print("X_test:\n", X_test)
    print("y_test:", y_test)
    print("\nPredictions:", y_pred)
    print(f"\nAccuracy: {accuracy:.2f}")

