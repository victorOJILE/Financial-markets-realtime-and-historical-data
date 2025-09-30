## Multinomial Logistic Regression
This is an extension of binary logistic regression that is used for classification problems where the dependent variable has three or more categories that have no inherent order (nominal).

It's more complex, training multiple binary models, but it's appropriate for unordered data or unquantitative significance like “disease A” vs “disease B” vs “disease C”.

In this case, the softmax function is used in place of the sigmoid function.

### .
### Usage Examples
- Predicting a customer's favorite cuisine from a list: "Italian," "Mexican," "Indian," or "Chinese." There is no natural order to these choices.

- Predicting a person's mode of transportation: "Car," "Bus," "Bike," or "Walking."

- Classifying a news article by topic: "Politics," "Sports," "Technology," or "Entertainment."

## .
### How it works
The core idea is to break down the multi-class problem into a series of binary logistic regression problems.

`Reference Category:` One category is chosen as the "reference" or "baseline" category. Let's say we choose "Car" as our reference for the transportation example.

`Separate Models:` The algorithm then trains a separate binary logistic regression model for each of the other categories against the reference category.

- Model 1: Predicts the probability of "Bus" versus "Car."
- Model 2: Predicts the probability of "Bike" versus "Car."
- Model 3: Predicts the probability of "Walking" versus "Car."

`Probabilities:` Each of these models outputs a probability, which is then run through a function called the Softmax function.

This function ensures that the probabilities of all categories (including the reference category) sum to 1.

`Final Prediction:` The model's final prediction for a new data point is the category with the highest calculated probability.

---
### .
### Example 1
This example will classify a dataset of flower species. The famous Iris dataset has three species: "setosa," "versicolor," and "virginica." 

These are non-ordered categories, making it a perfect use case for multinomial logistic regression.

```
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Load the dataset
# The Iris dataset is a classic for multi-class classification.
iris = load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Create and train the model
# The 'multi_class' parameter is key here.
# 'multinomial' tells the model to use the softmax function for multi-class classification.
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)

model.fit(X_train, y_train)

# 3. Predict the class for the test set.
y_pred = model.predict(X_test)

# 4. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("--- Multinomial Logistic Regression ---")
print(f"Model Accuracy: {accuracy:.4f}\n")

# 5. Let's predict the species of a new flower with these features:
# [sepal length, sepal width, petal length, petal width]
new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])

# Predict the class label (0, 1, or 2)
predicted_class = model.predict(new_flower)
print(f"Predicted class index: {predicted_class[0]}")
print(f"Predicted species: {class_names[predicted_class[0]]}")

# Predict the probabilities for each class
predicted_probas = model.predict_proba(new_flower)
print(f"Predicted probabilities for each species: {predicted_probas[0].round(4)}")
print(f"  - Setosa:     {predicted_probas[0][0]:.4f}")
print(f"  - Versicolor: {predicted_probas[0][1]:.4f}")
print(f"  - Virginica:  {predicted_probas[0][2]:.4f}")
```

### .
### Example 2
Below is an example of implementing multinomial logistic regression using the Digits dataset from scikit-learn:

```
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model, metrics

digits = datasets.load_digits()

X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

reg = linear_model.LogisticRegression(max_iter=10000, random_state=0)
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

print(f"Logistic Regression model accuracy: {metrics.accuracy_score(y_test, y_pred) * 100:.2f}%")
```
Output:

Logistic Regression model accuracy: 96.66%

This model is used to predict one of 10 digits (0-9) based on the image features.

The sklearn.linear_model.LogisticRegression class is versatile. By setting multi_class='multinomial', you tell it to use a softmax approach to handle the non-ordered categories.
