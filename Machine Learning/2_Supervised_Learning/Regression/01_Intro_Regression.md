## Regression Techniques in ML
Regression Analysis is a fundamental concept in machine learning used to model relationships between dependent and independent variables.

### .
### Types of Regression Techniques
`Linear Regression`

---
`Polynomial Regression`

This is an extension of linear regression and is used to model a non-linear relationship between the dependent variable and independent variables.

---
`Stepwise Regression`

Stepwise regression is used for fitting regression models with predictive models.
It is carried out automatically. 

With each step, the variable is added or subtracted from the set of explanatory variables. 

Here is the code for simple demonstration of the stepwise regression approach.
```
from sklearn.linear_model import StepwiseLinearRegression

model = StepwiseLinearRegression(forward=True,backward=True,
 verbose=1)

model.fit(X, y)
y_pred = model.predict(X_new)
```

---
`Decision Tree Regression`

A Decision Tree is the most powerful and popular tool for classification and prediction.

---
`Random Forest Regression`

Random Forest is an ensemble technique capable of performing both regression and classification tasks with the use of multiple decision trees and a technique called Bootstrap and Aggregation, commonly known as bagging. 

The basic idea behind this is to combine multiple decision trees in determining the final output rather than relying on individual decision trees. 

---
`Support Vector Regression (SVR)`

SVR tries to find a function that best predicts the continuous output value for a given input value.

SVR can use both linear and non-linear kernels. A linear kernel is a simple dot product between two input vectors, while a non-linear kernel is a more complex function that can capture more intricate patterns in the data.

Here is the code for simple demonstration of the Support vector regression approach.

```
from sklearn.svm import SVR

model = SVR(kernel='linear')
model.fit(X, y)

y_pred = model.predict(X_new)
```

---
`Ridge Regression`

This is a technique for analyzing multiple regression data. When multicollinearity occurs, least squares estimates are unbiased. 

This is a regularized linear regression model, it tries to reduce the model complexity by adding a penalty term to the cost function. A degree of bias is added to the regression estimates, and as a result, ridge regression reduces the standard errors.

Here is the code for simple demonstration of the Ridge regression approach.

```
from sklearn.linear_model import Ridge

model = Ridge(alpha=0.1)
model.fit(X, y)

y_pred = model.predict(X_new)
```

---
`Lasso Regression`
Lasso regression is a regression analysis method that performs both variable selection and regularization. Lasso regression uses soft thresholding. Lasso regression selects only a subset of the provided covariates for use in the final model.

This is another regularized linear regression model, it works by adding a penalty term to the cost function, but it tends to zero out some features' coefficients, which makes it useful for feature selection.

Here is the code for simple demonstration of the Lasso regression approach.

```
from sklearn.linear_model import Lasso

model = Lasso(alpha=0.1)
model.fit(X, y)

y_pred = model.predict(X_new)
```

---
`ElasticNet Regression`

Linear Regression suffers from overfitting and can’t deal with collinear data. 

When there are many features in the dataset and even some of them are not relevant to the predictive model. This makes the model more complex with a too-inaccurate prediction on the test set (or overfitting). 

Such a model with high variance does not generalize on the new data. So, to deal with these issues, we include both L-2 and L-1 norm regularization to get the benefits of both Ridge and Lasso at the same time. 

The resultant model has better predictive power than Lasso. It performs feature selection and also makes the hypothesis simpler.

```
from sklearn.linear_model import ElasticNet

model = ElasticNet(alpha=0.1, l1_ratio=0.5)
model.fit(X, y)

y_pred = model.predict(X_new)
```

---
`Bayesian Linear Regression`

As the name suggests this algorithm is purely based on Bayes Theorem. Because of this reason only we do not use the Least Square method to determine the coefficients of the regression model.

So, the technique which is used here to find the model weights and parameters relies on features posterior distribution and this provides an extra stability factor to the regression model which is based on this technique.

Here is the code for simple demonstration of the Bayesian Linear regression approach.

```
from sklearn.linear_model import BayesianLinearRegression

model = BayesianLinearRegression()
model.fit(X, y)

y_pred = model.predict(X_new)
```

### .
### FAQs
`What are the 2 main types of regression?`

- The two main types of regression are linear regression and logistic regression. 
- Linear regression is used to predict a continuous numerical outcome, 
- while logistic regression is used to predict a binary categorical outcome (e.g., yes or no, pass or fail).

#### .
`What are the two types of variables in regression?`

The two types of variables in regression are independent variables and dependent variables. 
- Independent variables are the inputs to the regression model, while the 
- dependent variable is the output that the model is trying to predict.

#### .
`Why is regression called regression?`

The term "regression" was coined by Sir Francis Galton in the late 19th century. He used the term to describe the phenomenon of children's heights tending to regress towards the mean of the population, meaning that taller-than-average parents tend to have children who are closer to the average height, and shorter-than-average parents tend to have children who are closer to the average height.

#### .
`How to calculate regression?`

There are many different ways to calculate regression, but the most common method is gradient descent.

Gradient descent is an iterative algorithm that updates the parameters of the regression model in the direction that minimizes the error between the predicted and actual values of the dependent variable.
