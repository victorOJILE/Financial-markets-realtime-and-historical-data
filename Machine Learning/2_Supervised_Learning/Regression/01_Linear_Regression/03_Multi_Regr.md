## Multiple Linear Regression
Multiple Linear Regression extends the concept of Linear regression by modelling the relationship between a dependent variable and two or more independent variables. 

The goal of the algorithm is to find the best fit line equation that can predict the values based on the independent variables. 

Think of it this way: a simple linear regression finds the "best-fit" line in a 2D space (x, y). Multiple regression, with two independent variables, finds the "best-fit" plane in a 3D space. 

---
A regression model learns from the dataset with known X and y values and uses it to predict y values for unknown X.

### .
### Multiple Regression Implementation 
Take a look at the data set below, it contains some information about cars.

|Car|Model|Volume|Weight|CO2|
|---|---|---|---|---|
|Toyota|Aygo|1000|790|99|
|Mitsubishi|Space Star|1200|1160|95|
|Skoda|Citigo|1000|929|95|
|Fiat|500|900|865|90|
|Mini|Cooper|1500|1140|105|
|VW|Up!|1000|929|105|
|Skoda|Fabia|1400|1109|90|
|Mercedes|A-Class|1500|1365|92|

---
We can predict the CO2 emission of a car based on the size of the engine, but with multiple regression we can throw in more variables, like the weight of the car, to make the prediction more accurate.

From the sklearn module we will use the LinearRegression() method to create a linear regression object.

#### Example
> Tip: It is common to name the list of independent values with a upper case X, and the list of dependent values with a lower case y.

```
import pandas
from sklearn import linear_model

df = pandas.read_csv("data.csv")

X = df[['Weight', 'Volume']] # independent values
y = df['CO2'] # dependent values

regr = linear_model.LinearRegression()
regr.fit(X, y)

# predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300cm3:
predictedCO2 = regr.predict([[2300, 1300]])

print(predictedCO2)
```

#### Result:
> [107.2087328]

We have predicted that a car with 1.3 liter engine, and a weight of 2300 kg, will release approximately 107 grams of CO2 for every kilometer it drives.

## .
### Coefficient
The coefficient is a factor that describes the relationship with an unknown variable.

**Example:** if x is a variable, then 2x is x two times. x is the unknown variable, and the number 2 is the coefficient.

In this case, we can ask for the coefficient value of weight against CO2, and for volume against CO2.

The answer(s) we get tells us what would happen if we increase, or decrease, one of the independent values.

---
#### Example
Print the coefficient values of the regression object:
```
print(regr.coef_)
```

#### Result:
> [0.00755095 0.00780526]


## .
#### Result Explained
The result array represents the coefficient values of weight and volume.

- Weight: 0.00755095
- Volume: 0.00780526

These values tell us that if the weight increase by 1kg, the CO2 emission increases by 0.00755095g.

And if the engine size (Volume) increases by 1cm3, the CO2 emission increases by 0.00780526g.

---
I think that is a fair guess, but let test it!

We have already predicted that if a car with a 1300cm3 engine weighs 2300kg, the CO2 emission will be approximately 107g.

---
What if we increase the weight with 1000kg?

#### Example
Copy the example from before, but change the weight from 2300 to 3300:
```
import pandas
from sklearn import linear_model

df = pandas.read_csv("data.csv")

X = df[['Weight', 'Volume']]
y = df['CO2']

regr = linear_model.LinearRegression()
regr.fit(X, y)

predictedCO2 = regr.predict([[3300, 1300]])

print(predictedCO2)
```

#### Result:
> [114.75968007]

We have predicted that a car with 1.3 liter engine, and a weight of 3300 kg, will release approximately 115 grams of CO2 for every kilometer it drives.

Which shows that the coefficient of 0.00755095 is correct:
> 107.2087328 + (1000 * 0.00755095) = 114.75968

### .
### Multicollinearity in Multiple LR
Multicollinearity arises when two or more independent variables are highly correlated with each other. This can make it difficult to find the individual contribution of each variable to the dependent variable.

To detect multicollinearity we can use:

`Correlation Matrix:` A correlation matrix helps to find relationships between independent variables. High correlations (close to 1 or -1) suggest multicollinearity.

`VIF (Variance Inflation Factor):` VIF quantifies how much the variance of a regression coefficient increases if predictors are correlated. A high VIF typically above 10 indicates multicollinearity.

### .
### Assumptions of Multiple Regression Model
Similar to simple linear regression we have some assumptions in multiple linear regression which are as follows:

`Linearity:` Relationship between dependent and independent variables should be linear.

`Homoscedasticity:` Variance of errors should remain constant across all levels of independent variables.

`Multivariate Normality:` Residuals should follow a normal distribution.

`No Multicollinearity:` Independent variables should not be highly correlated. Because you are dealing with multiple variables, it's important to select the right ones. 

Including too many irrelevant features can lead to a less efficient and potentially overfitted model. 

Techniques like forward selection, backward elimination, or regularization methods (which we can discuss later) are used to choose the best set of features.
