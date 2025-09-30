## Polynomial Regression
Polynomial Regression is a form of linear regression where the relationship between the independent variable (x) and the dependent variable (y) is modelled as an nth degree polynomial. 

### .
### Need for Polynomial Regression
`Non-linear Relationships:` Polynomial regression is used when the relationship between the independent variable (input) and dependent variable (output) is non-linear. 

Unlike linear regression which fits a straight line, it fits a polynomial equation to capture the curve in the data.

---
`Better Fit for Curved Data:` When a researcher hypothesizes a curvilinear relationship, polynomial terms are added to the model. 

A linear model often results in residuals with noticeable patterns which shows a poor fit. It can capture these non-linear patterns effectively.

---
`Flexibility and Complexity:` It does not assume all independent variables are independent. By introducing higher-degree terms, it allows for more flexibility and can model more complex, curvilinear relationships between variables.

#### .
While it looks like a curve when plotted, the core of polynomial regression is still a linear model in terms of its coefficients. 

It transforms the original feature into a higher-degree polynomial feature set (e.g., if the original feature is x, it might create new features like x², x³, etc.).

The algorithm's goal is to find the optimal coefficients for each of these polynomial terms. This allows the model to learn a more flexible, curved relationship that can better capture complex patterns in the data.

### .
### Example 1:
|Years of Experience|Salary (in dollars)|
|---|---|
|1|50,000|
|2|55,000|
|3|65,000|
|4|80,000|
|5|110,000|
|6|150,000|
|7|200,000|

Now, let's apply polynomial regression to model the relationship between years of experience and salary. We'll use a quadratic polynomial (degree 2) which includes both linear and quadratic terms for better fit.

To find the coefficients that minimize the difference between the predicted and actual salaries, we can use the Least Squares method. 

The objective is to minimize the sum of squared differences between the predicted salaries and the actual data points which allows us to fit a model that captures the non-linear progression of salary with respect to experience.

```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

datas = pd.read_csv('/content/data.csv')

X = datas.iloc[:, 1:2].values
y = datas.iloc[:, 2].values

# First fit a simple linear regression 
lin = LinearRegression()
lin.fit(X, y)

# Fitting the Polynomial Regression Model
# Use a polynomial of degree 4.

poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)

poly.fit(X_poly, y)

lin2 = LinearRegression()
lin2.fit(X_poly, y)

# Visualizing the Linear Regression Results
plt.scatter(X, y, color='blue')

plt.plot(X, lin.predict(X), color='red')
plt.title('Linear Regression')
plt.xlabel('Temperature')
plt.ylabel('Pressure')

plt.show()

# Visualize the Polynomial Regression Results
plt.scatter(X, y, color='blue')

plt.plot(X, lin2.predict(poly.fit_transform(X)),
 color='red')
plt.title('Polynomial Regression')
plt.xlabel('Temperature')
plt.ylabel('Pressure')

plt.show()
```
#### .
#### Predict New Results
To predict new values using both linear and polynomial regression we need to ensure the input variable is in a 2D array format.

```
pred = 110.0
predarray = np.array([[pred]])
lin.predict(predarray)
# array([0.20675333])


pred2 = 110.0
pred2array = np.array([[pred2]])
lin2.predict(poly.fit_transform(pred2array))
# array([0.43295877])
```

### .
### Example 2
We have registered 18 cars as they were passing a certain tollbooth.
We have registered the car's speed, and the time of day (hour) the passing occurred.

The x-axis represents the hours of the day and the y-axis represents the speed:

```
import numpy
import matplotlib.pyplot as plt

x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

# plt.scatter(x, y)
# plt.show()

mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))
myline = numpy.linspace(1, 22, 100)

plt.scatter(x, y)
plt.plot(myline, mymodel(myline))
plt.show()

mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))

# specify how the line will display, 
# we start at position 1, and end at position 22:

myline = numpy.linspace(1, 22, 100)

# Draw the line of polynomial regression:
plt.plot(myline, mymodel(myline))
```

### .
### R-Squared
It is important to know how well the relationship between the values of the x- and y-axis is, if there are no relationship the polynomial regression can not be used to predict anything.

- The relationship is measured with a value called the r-squared.

- The r-squared value ranges from 0 to 1, where 0 means no relationship, and 1 means 100% related.

Python and the Sklearn module will compute this value for you, all you have to do is feed it with the x and y arrays:

#### Example
How well does my data fit in a polynomial regression?
```
mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))

print(r2_score(y, mymodel(x)))
```
> Note: The result 0.94 shows that there is a very good relationship, and we can use polynomial regression in future predictions.

#### .
#### Predict Future Values
Now we can use the information we have gathered to predict future values.

Let us try to predict the speed of a car that passes the tollbooth at around the time 17:00:

To do so, we need the same mymodel array from the example above:

```
mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))

speed = mymodel(17)
print(speed)
```
The example predicted a speed to be **88.87**, which we also could read from the diagram

### .
### Overfitting and Underfitting
A key consideration with polynomial regression is the degree of the polynomial (e.g., a 2nd-degree polynomial has an x² term, a 3rd-degree has an x³ term, etc.).

---
Underfitting occurs when the model is too simple to capture the real patterns in the data. This usually happens with a low-degree polynomial.

---
Overfitting happens when the model is too complex and fits the training data too closely helps in making it perform poorly on new data.

A high-degree polynomial can be too flexible. It can bend and twist to pass very close to every single data point in the training set. 

While this results in a very low MSE on the training data, the model might be capturing noise and random fluctuations instead of the underlying trend.

---
To avoid this, we use techniques like Lasso and Ridge regression which helps to simplify the model by limiting the size of the coefficients.

The key is to choose the right polynomial degree to ensure the model is neither too complex nor too simple which helps it work well on both the training data and new data.

### .
### Bias Vs Variance Tradeoff
Bias Vs Variance Tradeoff helps us avoid both overfitting and underfitting by selecting the appropriate polynomial degree. 

As we increase the polynomial degree, the model fits the training data better but after a certain point, it starts to overfit. This is visible when the gap between training and validation errors begins to widen. 

The goal is to choose a polynomial degree where the model captures the data patterns without becoming too complex which ensures a good generalization.

### .
### Application of Polynomial Regression
`Modeling Growth Rates:` Polynomial regression is used to model non-linear growth rates such as the growth of tissues over time.

`Disease Epidemic Progression:` It helps track and predict the progression of disease outbreaks, capturing the non-linear nature of epidemic curves.

`Environmental Studies:` It is applied in studies like the distribution of carbon isotopes in lake sediments where relationships are non-linear.

`Economics and Finance:` It is used to analyze non-linear relationships in financial markets, predicting trends and fluctuations over time.
Advantages

`Fits a Wide Range of Curves:` Polynomial regression can model a broad range of non-linear relationships helps in making it versatile for complex data.

`Captures Non-linear Patterns:` It provides a more accurate approximation of relationships when data follows a curvilinear pattern, unlike linear regression.

`Flexible Modeling:` It can capture and represent the nuances in the data helps in making it ideal for situations where linear models fail.

### .
### Disadvantages
`Sensitivity to Outliers:` It is highly sensitive to outliers and a few extreme values can skew the results significantly.

`Overfitting:` With higher-degree polynomials, there’s a risk of overfitting where the model becomes too complex and fails to generalize well to new data.

`Limited Outlier Detection:` Unlike linear regression, it has fewer built-in methods for detecting or handling outliers helps in making it challenging to identify when the model is affected by them.
