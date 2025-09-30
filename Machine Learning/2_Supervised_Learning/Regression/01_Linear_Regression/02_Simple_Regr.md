## Simple Linear Regression
Python has methods for finding a relationship between data-points and to draw a line of linear regression. We will show you how to use these methods instead of going through the mathematic formula.

In the example below, the x-axis represents age, and the y-axis represents speed. We have registered the age and speed of 13 cars as they were passing a tollbooth. Let us see if the data we collected could be used in a linear regression:

#### Example
Start by drawing a scatter plot:

Import scipy and draw the line of Linear Regression:
```
import matplotlib.pyplot as plt
from scipy import stats

x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

plt.scatter(x, y)
# plt.show()

# or

slope, intercept, r, p, std_err = stats.linregress(x, y)

def myfunc(x):
  return slope * x + intercept

mymodel = list(map(myfunc, x))

# Draw the line of linear regression
plt.plot(x, mymodel)
plt.show()
```

#### Result:
![]()

## .
### Example Explained
Execute a method that returns some important key values of Linear Regression:
```
slope, intercept, r, p, std_err = stats.linregress(x, y)
```

---
Create a function that uses the *slope* and *intercept* values to return a new value. This new value represents where on the y-axis the corresponding x value will be placed:
```
def myfunc(x):
  return slope * x + intercept
```

---
Run each value of the x array through the function. This will result in a new array with new values for the y-axis:
```
mymodel = list(map(myfunc, x))
```

## .
## R for Relationship
It is important to know how the relationship between the values of the x-axis and the values of the y-axis is, if there are no relationship the linear regression can not be used to predict anything.

- This relationship - the coefficient of correlation - is called r.

- The r value ranges from -1 to 1, where 0 means no relationship, and 1 (and -1) means 100% related.

Python and the Scipy module will compute this value for you, all you have to do is feed it with the x and y values.

#### Example
How well does my data fit in a linear regression?

```
from scipy import stats

x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

slope, intercept, r, p, std_err = stats.linregress(x, y)

print(r)
```

> **Note:** The result -0.76 shows that there is a relationship, not perfect, but it indicates that we could use linear regression in future predictions.

## .
### Predict Future Values
Now we can use the information we have gathered to predict future values.

#### Example: 
Let us try to predict the speed of a 10 years old car.

To do so, we need the same myfunc() function from the example above:

```
from scipy import stats

x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

slope, intercept, r, p, std_err = stats.linregress(x, y)

def myfunc(x):
  return slope * x + intercept

speed = myfunc(10)

print(speed)
```
The example predicted a speed at 85.6, which we also could read from the diagram:

![Scatter Plot, Predicted speed at 85.6](./media/predicted_speed_scatter.jpg)

## .
### Logic in Machine Learning
The core logic of linear regression is to find the "best-fit" straight line that minimizes the error between the model's predictions and the actual data points.

`The Goal:` The algorithm's primary objective is to find the optimal values for the slope and the y-intercept of this line. These values define the line that best represents the data.

`Minimizing Error:` To determine what "best-fit" means, a "cost function" (or "loss function") is used. A common one is the Mean Squared Error (MSE), which calculates the average of the squared differences between the predicted values and the actual values. The squaring of errors is important because it makes all differences positive and gives more weight to larger errors (outliers).

`Optimization:` The model uses an optimization algorithm, like Gradient Descent, to iteratively adjust the slope and intercept values in a direction that reduces the cost function. This process continues until the error is as small as possible, resulting in the line that best fits the training data.

## .
### Best Fit
In linear regression, "best fit" refers to the single straight line that most accurately represents the trend in the data. The goal is to find the line that minimizes the total error between the predicted values and the actual data points.

It's what differentiates it from simply drawing an arbitrary line through a dataset.

### .
#### Positive Linear Regression Line
A positive linear regression line indicates a direct relationship between the independent variable (X) and the dependent variable (Y). 

This means that as the value of X increases, the value of Y also increases. The slope of a positive linear regression line is positive, meaning that the line slants upward from left to right.

#### Negative Linear Regression Line
A negative linear regression line indicates an inverse relationship between the independent variable (X) and the dependent variable (Y ). 

This means that as the value of X increases, the value of Y decreases. The slope of a negative linear regression line is negative, meaning that the line slants downward from left to right.

### .
#### Residuals (Errors)
The error for a single data point is called a residual. It's the vertical distance between the data point and the regression line. A positive residual means the data point is above the line, and a negative residual means it's below.

#### The Problem with Summing Errors
You can't just sum all the residuals because positive and negative errors would cancel each other out, potentially leading to a "best fit" line that is clearly wrong but has a total sum of residuals close to zero.

#### The Solution
> Mean Squared Error (MSE)

To overcome this, linear regression uses a cost function, most commonly the Mean Squared Error (MSE). The algorithm calculates the square of each residual, sums them up, and then takes the average.

Squaring the residuals ensures that all errors are positive, so they don't cancel each other out.

Squaring also gives a much higher penalty to larger errors (outliers), forcing the line to be closer to all the data points, rather than ignoring a few far-away ones.

---
Therefore, the "best fit" line is the one whose slope and intercept values result in the lowest possible MSE. 

An optimization algorithm like Gradient Descent is used to find these optimal values. It iteratively adjusts the line's parameters, measuring the change in MSE at each step, and moving in the direction that reduces the error.


## .
### How and where it is used:
#### Training the Model
When you "train" a regression model, the algorithm analyzes your training data (the set of input features and corresponding output values) and calculates the optimal coefficients. These coefficients mathematically define the best-fit line or curve.

#### The Prediction Phase
Once the model is trained, you can give it a new set of input features for which you don't know the output value. The model will then use the equation of its best-fit line or curve to calculate and predict the corresponding output.

#### .
#### Example
`Predicting House Prices`

Imagine you have a linear regression model trained on data of house prices and their sizes. The model has determined the best-fit line, which can be represented by a simple equation like:

> Price = (Slope × Size) + Intercept

---
Let's say the model found a slope of 150 and an intercept of 50,000. The equation of the best-fit line is:

> Price = (150 × Size) + 50000

Now, you have a new house that is 2,000 square feet, and you want to predict its price. You would use the model's equation:

> PredictedPrice = (150 × 2000) + 50000 = 300000 + 50000 = 350000

The "best-fit" line gives you a predicted price of **$350,000**.

---
In essence, the "best-fit" line or curve is the mathematical function that the regression algorithm has learned from the data. This function is the core of the model, and it's what allows it to take new inputs and generate predictions. The visual representation of the line or curve is simply a way to understand the relationship that the model has learned.