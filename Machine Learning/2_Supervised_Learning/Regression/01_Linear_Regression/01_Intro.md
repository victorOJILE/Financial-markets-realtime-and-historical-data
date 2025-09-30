## Linear Regression in ML
The term regression is used when you try to find the relationship between variables.

In Machine Learning, and in statistical modeling, that relationship is used to predict the outcome of future events.

---
It uses the relationship between a dependent variable (the target you're trying to predict) and one or more independent variables (the features) by fitting a straight line to the data (meaning the output changes at a constant rate as the input changes).

This line can be used to predict future values.

---
Linear Regression is a supervised ML algorithm that learns from the labelled datasets and maps the data points with most optimized linear functions which can be used for prediction on new datasets. 

#### .
For example we want to predict a student's exam score based on how many hours they studied. We observe that as students study more hours, their scores go up. In the example of predicting exam scores based on hours studied. 

#### Here
- Independent variable (input): Hours studied because it's the factor we control or observe.
- Dependent variable (output): Exam score because it depends on how many hours were studied.

We use the independent variable to predict the dependent variable.

### .
### Importance of Linear Regression.
`Simplicity and Interpretability:` It’s easy to understand and interpret, making it a starting point for learning about machine learning.

---
`Predictive Ability:` Helps predict future outcomes based on past data, making it useful in various fields like finance, healthcare and marketing.

---
`Basis for Other Models:` Many advanced algorithms, like logistic regression or neural networks, build on the concepts of linear regression.

---
`Efficiency:` It’s computationally efficient and works well for problems with a linear relationship.

---
`Widely Used:` It’s one of the most widely used techniques in both statistics and machine learning for regression tasks.

---
`Analysis:` It provides insights into relationships between variables (e.g., how much one variable influences another).

### .
### Real world usage
Linear regression is widely used in various fields due to its simplicity and interpretability.

`Financial Forecasting:` Companies can use it to predict sales, stock prices, or customer lifetime value based on factors like advertising spend, historical data, and market trends.

`Predictive Modeling:` It can be used to predict house prices based on features like square footage, number of bedrooms, and location.

`Medical Research:` Researchers might use it to study the relationship between drug dosage and blood pressure or to predict a patient's risk of disease based on their age, weight, and other health indicators.

`Initial Analysis:` It often serves as a baseline model for more complex problems to understand the basic trends and relationships in the data.

### .
### Evaluation Metrics for LR
A variety of evaluation measures can be used to determine the strength of any linear regression model. These assessment metrics often give an indication of how well the model is producing the observed outputs.

The most common measurements are:

`Mean Square Error (MSE)`

Mean Squared Error (MSE) is an evaluation metric that calculates the average of the squared differences (residuals) between the actual and predicted values for all the data points. 

The difference is squared to ensure that negative and positive differences don't cancel each other out.

MSE is a way to quantify the accuracy of a model's predictions. MSE is sensitive to outliers as large errors contribute significantly to the overall score.

---
`Mean Absolute Error (MAE)`

Mean Absolute Error is an evaluation metric used to calculate the accuracy of a regression model. MAE measures the average absolute difference between the predicted values and actual values.

Lower MAE value indicates better model performance. It is not sensitive to the outliers as we consider absolute differences.

---
`Coefficient of Determination (R-squared)`

R-Squared is a statistic that indicates how much variation the developed model can explain or capture. It is always in the range of 0 to 1. In general, the better the model matches the data, the greater the R-squared number.

- Residual sum of Squares(RSS): The sum of squares of the residual for each data point in the plot or data is known as the residual sum of squares or RSS. It is a measurement of the difference between the output that was observed and what was anticipated.

- Total Sum of Squares (TSS): The sum of the data points' errors from the answer variable's mean is known as the total sum of squares or TSS.

R squared metric is a measure of the proportion of variance in the dependent variable that is explained the independent variables in the model.

`Adjusted R-Squared Error`

Adjusted R^2 measures the proportion of variance in the dependent variable that is explained by independent variables in a regression model. Adjusted R-square accounts the number of predictors in the model and penalizes the model for including irrelevant predictors that don't contribute significantly to explain the variance in the dependent variables.

R2 is coeeficient of determination
Adjusted R-square helps to prevent overfitting. It penalizes the model with additional predictors that do not contribute significantly to explain the variance in the dependent variable.

While evaluation metrics help us measure the performance of a model, regularization helps in improving that performance by addressing overfitting and enhancing generalization.

### .
#### Regularization Techniques for Linear Models
`Lasso Regression (L1 Regularization)`

Lasso Regression is a technique used for regularizing a linear regression model, it adds a penalty term to the linear regression objective function to prevent overfitting.

---
`Ridge Regression (L2 Regularization)`

Ridge regression is a linear regression technique that adds a regularization term to the standard linear objective. Again, the goal is to prevent overfitting by penalizing large coefficient in linear regression equation. It useful when the dataset has multicollinearity where predictor variables are highly correlated.

---
`Elastic Net Regression`
Elastic Net Regression is a hybrid regularization technique that combines the power of both L1 and L2 regularization in linear regression objective.

### .
### Advantages of Linear Regression
Linear regression is a relatively simple algorithm, making it easy to understand and implement. 

The coefficients of the linear regression model can be interpreted as the change in the dependent variable for a one-unit change in the independent variable, providing insights into the relationships between variables.

---
Linear regression is computationally efficient and can handle large datasets effectively. It can be trained quickly on large datasets, making it suitable for real-time applications.

---
Linear regression is relatively robust to outliers compared to other machine learning algorithms. Outliers may have a smaller impact on the overall model performance.

---
Linear regression often serves as a good baseline model for comparison with more complex machine learning algorithms.

### .
### Disadvantages of Linear Regression
Linear regression assumes a linear relationship between the dependent and independent variables. If the relationship is not linear, the model may not perform well.

---
Linear regression is sensitive to multicollinearity, which occurs when there is a high correlation between independent variables. 

Multicollinearity can inflate the variance of the coefficients and lead to unstable model predictions.

---
Linear regression assumes that the features are already in a suitable form for the model. 

Feature engineering may be required to transform features into a format that can be effectively used by the model.

---
Linear regression is susceptible to both overfitting and underfitting. 

Overfitting occurs when the model learns the training data too well and fails to generalize to unseen data. 

Underfitting occurs when the model is too simple to capture the underlying relationships in the data.

---
Linear regression provides limited explanatory power for complex relationships between variables.

More advanced machine learning techniques may be necessary for deeper insights.
