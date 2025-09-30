# Binomial Logistics Regression
It is used for binary classification where the output can be one of two possible categories such as Yes/No, True/False or 0/1. 

In binomial logistic regression, the target variable can only have two possible values such as "0" or "1", "pass" or "fail". The sigmoid function is used for prediction.

An example of which is predicting if a tumor is malignant or benign.

#### Example 1
```
from sklearn.linear_model import LogisticRegression

# X represents the size of a tumor in centimeters.
# Reshaped into a column from a row 
# for the LogisticRegression() function to work.

X = numpy.array([3.78, 2.44, 2.09, 0.14, 1.72, 
  1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1,1)

# y represents whether or not the tumor 
# is cancerous (0 for "No", 1 for "Yes")

y = numpy.array([0, 0, 0, 0, 0, 
  0, 1, 1, 1, 1, 1, 1])

model = LogisticRegression()
model.fit(X,y)

# predict if tumor is cancerous 
# where the size is 3.46mm:

predicted = model.predict(numpy.array([3.46]).reshape(-1,1))
print(predicted)
```

#### Result
> [0]

We have predicted that a tumor with a size of 3.46mm will not be cancerous.

---
#### Results Explained
- 3.78 0.61 The probability that a tumor with the size 3.78cm is cancerous is 61%.

- 2.44 0.19 The probability that a tumor with the size 2.44cm is cancerous is 19%.

- 2.09 0.13 The probability that a tumor with the size 2.09cm is cancerous is 13%.

## .
### Coefficient
In logistic regression, the coefficient is the expected change in log-odds of having the outcome per unit change in X.

This does not have the most intuitive understanding so let's use it to create something that makes more sense, odds.
```
log_odds = model.coef_
odds = numpy.exp(log_odds)

print(odds)
```

#### Result
> [4.03541657]
 
This tells us that as the size of a tumor increases by 1mm the odds of it being a cancerous tumor increases by 4x.

## .
### Probability
The coefficient and intercept values can be used to find the probability that each tumor is cancerous.

Create a function that uses the model's coefficient and intercept values to return a new value. This new value represents probability that the given observation is a tumor:

```
def logit2prob(model, x):
  log_odds = model.coef_ * x + model.intercept_
  
  # To convert to odds, exponentiate it
  odds = numpy.exp(log_odds)
  
  # convert to probability
  probability = odds / (1 + odds)
  
  return(probability)
```

Let us now use the function with what we have learned to find out the probability that each tumor is cancerous.

```
print(logit2prob(model, X))
```

### .
### Example 2
We will be using sckit-learn library for this and shows how to use the breast cancer dataset to implement a Logistic Regression model for classification.

```
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=23)

clf = LogisticRegression(max_iter=10000, random_state=0)
clf.fit(X_train, y_train)

acc = accuracy_score(y_test, clf.predict(X_test)) * 100
print(f"Logistic Regression model accuracy: {acc:.2f}%")
```
Output:
> Logistic Regression model accuracy (in %): 96.49%

This code uses logistic regression to classify whether a sample from the breast cancer dataset is malignant or benign.

## .
### Where Binomial Regression is Used
`Spam Detection:` Classifying emails as "spam" or "not spam."

`Medical Diagnosis:` Predicting whether a patient has a specific disease based on their symptoms, test results, and medical history.

`Credit Scoring:` Predicting the probability of a loan applicant defaulting on a loan ("default" or "no default").

`Customer Churn:` Determining whether a customer is likely to cancel a subscription ("churn" or "not churn").

`Marketing:` Predicting whether a user will click on an ad or not.
