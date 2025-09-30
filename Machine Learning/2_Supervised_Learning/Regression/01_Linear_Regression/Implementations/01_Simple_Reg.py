"""
Utility Functions in Univariate Linear Regression Model
  Prediction with linear regression 
  Cost function 
  Gradient Descent For Parameter Estimation
  Update Coefficients
  Stop Iterations
"""

class LinearRegressor:
 def __init__(self, x, y, alpha=0.01, b0=0, b1=0):
  """ 
  x: input feature
  y: result / target
  alpha: learning rate, default is 0.01
  b0, b1: linear regression coefficient.
  """
  self.x = x
  self.y = y
  self.alpha = alpha
  self.b0 = b0
  self.b1 = b1
  if len(x) != len(y):
   raise TypeError("""x and y should have same number of rows.""")
    
 def cost_derivative(x, y, b0, b1):
  """
  The cost function computes the error with the current value of regression coefficients. It quantitatively defines how far the model predicted value is from the actual value wrt regression coefficients which have the lowest rate of error. 
  
  Mean-Squared Error(MSE) = sum of squares of difference between predicted and actual value
  
  We use square so that positive and negative error does not cancel out each other.

  Here:

  y is listed of expected values 
  x is the independent variable 
  b0 and b1 are regression coefficient 
  """
  errors = []
  for x, y in zip(x, y):
   prediction = predict(x, b0, b1)
   expected = y
   difference = prediction-expected
   errors.append(difference)
  mse = sum([error * error for error in errors])/len(errors)
  return mse
  
 def predict(model, x):
  """
  Predict the value of y on a given value of x by multiplying and adding the coefficient of regression to the input x.
  """
  return model.b0 + model.b1 * x
 
 def grad_fun(model, i):
  """
  Gradient Descent
  We take the partial derivative of the cost function wrt to our regression coefficient
  and multiply with the learning rate alpha 
  and subtract it from our coefficient to adjust our regression coefficient.
  """
  x, y, b0, b1 = model.x, model.y, model.b0, model.b1
  predict = model.predict
  return sum([
   2 * (predict(xi) - yi) * 1
   if i == 0
   else (predict(xi) - yi) * xi
   for xi, yi in zip(x, y)
  ]) / len(x)
 
 def fit(model):
  cost_derivative = model.cost_derivative
  i = 0
  max_epochs = 1000
  while i < max_epochs:
   i += 1 # Iteration count
  """
  At each iteration (epoch), the values of the regression coefficient are updated by a specific value wrt to the error from the previous iteration. 
  
  This updation is very crucial and is the crux of the machine learning applications that you write. 
  
  Updating the coefficients is done by penalizing their value with a fraction of the error that its previous values caused. 
  
  This fraction is called the learning rate. This defines how fast our model reaches to point of convergence(the point where the error is ideally 0).
  """
   model.b0 -= model.alpha * cost_derivative(0)
   model.b1 -= model.alpha * cost_derivative(1)

# Usage
linearRegressor = LinearRegressor(
 x=[i for i in range(12)],
 y=[2 * i + 3 for i in range(12)],
 alpha=0.03
)

linearRegressor.fit()
print(linearRegressor.predict(12))
# Output: 27.00000004287766


class LinearRegressor:
 def __init__(self, x, y, alpha=0.01, b0=0, b1=0):
  self.x = x
  self.y = y
  self.alpha = alpha
  self.b0 = b0
  self.b1 = b1
  if len(x) != len(y):
   raise TypeError("""x and y should have same number of rows.""")
    
 def cost_derivative(x, y, b0, b1):
  errors = []
  for x, y in zip(x, y):
   prediction = predict(x, b0, b1)
   expected = y
   difference = prediction-expected
   errors.append(difference)
  mse = sum([error * error for error in errors])/len(errors)
  return mse
  
 def predict(model, x):
  return model.b0 + model.b1 * x
 
 def grad_fun(model, i):
  x, y, b0, b1 = model.x, model.y, model.b0, model.b1
  predict = model.predict
  return sum([
   2 * (predict(xi) - yi) * 1
   if i == 0
   else (predict(xi) - yi) * xi
   for xi, yi in zip(x, y)
  ]) / len(x)
 
 def fit(model):
  cost_derivative = model.cost_derivative
  i = 0
  max_epochs = 1000
  while i < max_epochs:
   i += 1 # Iteration count
   model.b0 -= model.alpha * cost_derivative(0)
   model.b1 -= model.alpha * cost_derivative(1)
