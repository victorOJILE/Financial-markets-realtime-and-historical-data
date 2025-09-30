# Logistic Regression
This is a supervised machine learning algorithm used for classification not regression problems. 

Its goal is to predict a discrete, categorical outcome—like "yes" or "no," "spam" or "not spam," "benign" or "malignant."

It uses sigmoid function to convert inputs into a probability value between 0 and 1. 

For example, if the calculated probability is 0.7, and the threshold is 0.5, the model would classify the outcome as "yes" or "1."

### .
### Types of Logistic Regression
Logistic regression can be classified into three main types based on the nature of the dependent variable:

`Binomial Logistic Regression:` This type is used when the dependent variable has only two possible categories. Examples include Yes/No, Pass/Fail or 0/1.

`Multinomial Logistic Regression:` This is used when the dependent variable has three or more possible categories that are not ordered. For example, classifying animals into categories like "cat," "dog" or "sheep." 

It extends the binary logistic regression to handle multiple classes.

`Ordinal Logistic Regression:` This type applies when the dependent variable has three or more categories with a natural order or ranking. Examples include ratings like "low," "medium" and "high." 

It takes the order of the categories into account when modeling.

### .
### Assumptions of Logistic Regression
Understanding the assumptions behind logistic regression is important to ensure the model is applied correctly, main assumptions are:

`Independent observations:` Each data point is assumed to be independent of the others means there should be no correlation or dependence between the input samples.

`Binary dependent variables:` It takes the assumption that the dependent variable must be binary, means it can take only two values. For more than two categories SoftMax functions are used.

`Linearity relationship between independent variables and log odds:` The model assumes a linear relationship between the independent variables and the log odds of the dependent variable which means the predictors affect the log odds in a linear way.

`No outliers:` The dataset should not contain extreme outliers as they can distort the estimation of the logistic regression coefficients.

`Large sample size:` It requires a sufficiently large sample size to produce reliable and stable results.

### .
### Terminologies involved in Logistic Regression
`Independent Variables:` These are the input features or predictor variables used to make predictions about the dependent variable.

`Dependent Variable:` This is the target variable that we aim to predict. In logistic regression, the dependent variable is categorical.

`Logistic Function:` This function transforms the independent variables into a probability between 0 and 1 which represents the likelihood that the dependent variable is either 0 or 1.

`Odds:` This is the ratio of the probability of an event happening to the probability of it not happening. It differs from probability because probability is the ratio of occurrences to total possibilities.

`Log-Odds (Logit):` The natural logarithm of the odds. In logistic regression, the log-odds are modeled as a linear combination of the independent variables and the intercept.

`Coefficient:` These are the parameters estimated by the logistic regression model which shows how strongly the independent variables affect the dependent variable.

`Intercept:` The constant term in the logistic regression model which represents the log-odds when all independent variables are equal to zero.

`Maximum Likelihood Estimation (MLE):` This method is used to estimate the coefficients of the logistic regression model by maximizing the likelihood of observing the given data.

### .
### Evaluating LR Model?
Evaluating the logistic regression model helps assess its performance and ensure it generalizes well to new, unseen data. The following metrics are commonly used:

`Accuracy:` Accuracy provides the proportion of correctly classified instances.

`Precision:` Precision focuses on the accuracy of positive predictions.

`Recall (Sensitivity or True Positive Rate):` Recall measures the proportion of correctly predicted positive instances among all actual positive instances.

`F1 Score:` F1 score is the harmonic mean of precision and recall.

`Area Under the Receiver Operating Characteristic Curve (AUC-ROC):` The ROC curve plots the true positive rate against the false positive rate at various thresholds. 

AUC-ROC measures the area under this curve which provides an aggregate measure of a model's performance across different classification thresholds.

`Area Under the Precision-Recall Curve (AUC-PR):` Similar to AUC-ROC, AUC-PR measures the area under the precision-recall curve helps in providing a summary of a model's performance across different precision-recall trade-offs.

### .
### Understanding Sigmoid Function
The sigmoid function is a important part of logistic regression which is used to convert the raw output of the model into a probability value between 0 and 1.

---
This function takes any real number and maps it into the range 0 to 1 forming an "S" shaped curve called the sigmoid curve or logistic curve. 

Because probabilities must lie between 0 and 1, the sigmoid function is perfect for this purpose.

In logistic regression, we use a threshold value usually 0.5 to decide the class label.

---
If the sigmoid output is same or above the threshold, the input is classified as Class 1.

If it is below the threshold, the input is classified as Class 0.

This approach helps to transform continuous input values into meaningful class predictions.

## .
### The "Best Fit" and Cost Function
The concept of "best fit" is still about minimizing error, but the cost function used is different from MSE. 

Logistic regression uses a cost function called log loss or cross-entropy loss. This function heavily penalizes the model when it is confident in a wrong prediction. 

For example, if the actual outcome is "yes" but the model predicts a very low probability (close to 0), the log loss will be a very large number.
