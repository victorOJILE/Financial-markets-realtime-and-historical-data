# Grid Search
The majority of machine learning models contain parameters that can be adjusted to vary how the model learns. 

For example, the logistic regression model, from sklearn, has a parameter C that controls regularization,which affects the complexity of the model.

How do we pick the best value for C? The best value is dependent on the data used to train the model.

## .
### How does it work?
One method is to try out different values and then pick the value that gives the best score. This technique is known as a grid search. 

If we had to select the values for two or more parameters, we would evaluate all combinations of the sets of values thus forming a grid of values.

#### .

Before we get into the example it is good to know what the parameter we are changing does. Higher values of C tell the model, the training data resembles real world information, place a greater weight on the training data. While lower values of C do the opposite.

### Using Default Parameters
First let's see what kind of results we can generate without a grid search using only the base parameters.

#### Example 
```
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()

X = iris['data']
y = iris['target']

# default value for C in a logistic regression model is 1
# setting max_iter to a higher value 
# to ensure that the model finds a result.
logit = LogisticRegression(max_iter = 10000)

print(logit.fit(X,y))

print(logit.score(X,y))
```

With the default setting of C = 1, we achieved a score of 0.973.

Let's see if we can do any better by implementing a grid search with difference values of 0.973.

## .
### Implementing Grid Search
We will follow the same steps of before except this time we will set a range of values for C.

Knowing which values to set for the searched parameters will take a combination of domain knowledge and practice.

Since the default value for C is 1, we will set a range of values surrounding it.

> C = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]

#### Example
```
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()

X = iris['data']
y = iris['target']

logit = LogisticRegression(max_iter = 10000)

C = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]

scores = []

for choice in C:
  logit.set_params(C=choice)
  logit.fit(X, y)
  scores.append(logit.score(X, y))

print(scores)
```

## .
### Results Explained
We can see that the lower values of C performed worse than the base parameter of 1. However, as we increased the value of C to 1.75 the model experienced increased accuracy.

It seems that increasing C beyond this amount does not help increase model accuracy.

## .
### Note on Best Practices
We scored our logistic regression model by using the same data that was used to train it. If the model corresponds too closely to that data, it may not be great at predicting unseen data. This statistical error is known as over fitting.

To avoid being misled by the scores on the training data, we can put aside a portion of our data and use it specifically for the purpose of testing the model. Refer to the lecture on train/test splitting to avoid being misled and overfitting.