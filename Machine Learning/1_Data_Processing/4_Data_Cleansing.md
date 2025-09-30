## Data cleaning
This is an important step in the machine learning (ML) pipeline as it involves identifying and removing any missing duplicate or irrelevant data.

The goal of data cleaning is to ensure that the data is accurate, consistent and free of errors as raw data is often noisy, incomplete and inconsistent which can negatively impact the accuracy of model and its reliability of insights derived from it. 

Professional data scientists usually invest a large portion of their time in this step because of the belief that:
> “Better data beats fancier algorithms”

---
Clean datasets also helps in EDA that enhance the interpretability of data so that right actions can be taken based on insights.

### .
### How to Perform Data Cleanliness?
The process begins by thorough understanding data and its structure to identify issues like missing values, duplicates and outliers. 

#### .
Performing data cleaning involves a systematic process to identify and remove errors in a dataset.

`1. Removal of Unwanted Observations:` Identify and remove irrelevant or redundant (unwanted) observations from the dataset. 

This step involves analyzing data entries for duplicate records, irrelevant information or data points that do not contribute to analysis and prediction.

Removing them from dataset helps reducing noise and improving the overall quality of dataset.

`2. Fixing Structure errors:` Address structural issues in the dataset such as inconsistencies in data formats or variable types. 

Standardize formats ensure uniformity in data structure and hence data consistency.

`3. Managing outliers:` Outliers are those points that deviate significantly from dataset mean.

Identifying and managing outliers significantly improve model accuracy as these extreme values influence analysis.

Depending on the context decide whether to remove outliers or transform them to minimize their impact on analysis.

`4. Handling Missing Data:` To handle missing data effectively we need to impute missing values based on statistical methods, removing records with missing values or employing advanced imputation techniques. 

Handling missing data helps preventing biases and maintaining the integrity of data.

#### .
Throughout the process documentation of changes is crucial for transparency and future reference. Iterative validation is done to test effectiveness of the data cleaning resulting in a refined dataset and can be used for meaningful analysis and insights.

### .
### Database Cleaning in Python
Let's understand each step for Database Cleaning, using titanic dataset.
```
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('titanic.csv')
df.head()
```
**Output:**
>  

### .
### Data Inspection and Exploration
Let's first understand the data by inspecting its structure and identifying missing values, outliers and inconsistencies and check the duplicate rows with below python code:

```
df.duplicated()
```
**Output:**
>  

**Check the data information using df.info()**

```
df.info()
```
**Output:**
>  

From the above data info we can see that Age and Cabin have an **unequal number of counts**. And some of the columns are categorical and have data type objects and some are integer and float values.

**Check the Categorical and Numerical Columns.**
```
# Categorical columns
cat_col = [col for col in df.columns if df[col].dtype == 'object']

print('Categorical columns :',cat_col)

# Numerical columns
num_col = [col for col in df.columns if df[col].dtype != 'object']

print('Numerical columns :',num_col)
```
**Output:**
> Categorical columns: ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']

> Numerical columns : ['PassengerId', 'Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

**Check the total number of Unique Values in the Categorical Columns**
```
df[cat_col].nunique()
```
**Output:**
| | |
|---|---|
|Name|891|
|Sex|2|
|Ticket|681|
|Cabin|147|
|Embarked|3|
dtype: int64

### .
### Removal of Unwanted Observations
**Duplicate observations** most frequently arise during data collection and irrelevant observations are those that don’t actually fit with the specific problem that we’re trying to solve. 
- **Redundant observations** alter the efficiency to a great extent as the data repeats and may add towards the correct side or towards the incorrect side, therefore producing useless results.
- **Irrelevant observations** are any type of data that is of no use to us and can be removed directly.

Now we have to make a decision according to the subject of analysis which factor is important for our discussion.

As we know our machines don't understand the text data. So we have to either drop or convert the categorical column values into numerical types. 

Here we are dropping the Name columns because the Name will be always unique and it hasn't a great influence on target variables. 

For the ticket, Let's first print the 50 unique tickets.

```
df['Ticket'].unique()[:50]
```
**Output:**
>  

From the above tickets, we can observe that it is made of two like first values **'A/5 21171'** is joint from of **'A/5'** and **'21171'** this may influence our target variables. 

It will be the case of Feature Engineering. where we derived new features from a column or a group of columns. In the current case, we are dropping the **"Name"** and **"Ticket"** columns.

**Drop Name and Ticket Columns**

```
df1 = df.drop(columns=['Name','Ticket'])
df1.shape
```
**Output:**
> (891, 10)

### .
### Handling Missing Data
Missing data is a common issue in real-world datasets and it can occur due to various reasons such as human errors, system failures or data collection issues. 

Various techniques can be used to handle missing data, such as imputation, deletion or substitution.

Let's check the missing values columns-wise for each row using 
- df.isnull(): checks whether the values are null or not and returns boolean values 
- sum(): sum the total number of null values rows 
- Then, we divide it by the total number of rows present in the dataset
- Then multiply to get values in i.e per 100 values how much values are null.

```
round((df1.isnull().sum() / df1.shape[0]) * 100, 2)
```
**Output:**
| | |
|---|---|
|PassengerId|0.00|
|Survived|0.00|
|Pclass|0.00|
|Sex|0.00|
|Age|19.87|
|SibSp|0.00|
|Parch|0.00|
|Fare|0.00|
|Cabin|77.10|
|Embarked|0.22|

dtype: float64

---
We cannot just ignore or remove the missing observation. They must be handled carefully as they can be an indication of something important. 

- The fact that the value was missing may be informative in itself.
- In the real world we often need to make predictions on new data even if some of the features are missing!

As we can see from the above result that Cabin has 77% null values and Age has 19.87% and Embarked has 0.22% of null values.

So, it's not a good idea to fill 77% of null values. So we will drop the Cabin column. Embarked column has only 0.22% of null values so, we drop the null values rows of Embarked column.

```
df2 = df1.drop(columns='Cabin')
df2.dropna(subset=['Embarked'], axis=0, inplace=True)
df2.shape
```
**Output:**
> (889, 9)

---
Imputing the missing values from past observations.

- Again "missingness" is almost informative in itself and we should tell our algorithm if a value was missing.
- Even if we build a model to impute our values we’re not adding any real information. we’re just reinforcing the patterns already provided by other features. We can use **Mean imputation** or **Median imputations** for the case.

#### Note: 
- Mean imputation is suitable when the data is normally distributed and has no extreme outliers.
- Median imputation is preferable when the data contains outliers or is skewed.

```
# Mean imputation
df3 = df2.fillna(df2.Age.mean())
# Let's check the null values again
df3.isnull().sum()
```
**Output:**
>  

### .
### Handling Outliers
Outliers are extreme values that deviate significantly from the majority of the data. 

They can negatively impact the analysis and model performance. Techniques such as clustering, interpolation or transformation can be used to handle outliers.

---
To check the outliers we generally use a box plot. 

A box plot is a graphical representation of a dataset's distribution. It shows a variable's median, quartiles and potential outliers. The line inside the box denotes the median while the box itself denotes the interquartile range (IQR). 

The box plot extend to the most extreme non-outlier values within 1.5 times the IQR. Individual points beyond the box are considered potential outliers. 

A box plot offers an easy-to-understand overview of the range of the data and makes it possible to identify outliers or skewness in the distribution.

Let's plot the box plot for Age column data.

```
import matplotlib.pyplot as plt

plt.boxplot(df3['Age'], vert=False)
plt.ylabel('Variable')
plt.xlabel('Age')
plt.title('Box Plot')
plt.show()
```
**Output:**
>  

As we can see from the above Box and whisker plot, Our age dataset has outliers values. The values less than 5 and more than 55 are outliers.

```
# calculate summary statistics
mean = df3['Age'].mean()
std  = df3['Age'].std()

# Calculate the lower and upper bounds
lower_bound = mean - std*2
upper_bound = mean + std*2

print('Lower Bound :',lower_bound)
print('Upper Bound :',upper_bound)

# Drop the outliers
df4 = df3[(df3['Age'] >= lower_bound) 
  & (df3['Age'] <= upper_bound)]
```
**Output:**
> Lower Bound : 3.705400107925648

> Upper Bound : 55.578785285332785

Similarly, we can remove the outliers of the remaining columns.

### .
### Data Transformation 
This involves converting the data from one form to another to make it more suitable for analysis. Techniques such as normalization, scaling or encoding can be used to transform the data.

### Data validation and verification
These involve ensuring that the data is accurate and consistent by comparing it with external sources or expert knowledge. 

For the machine learning prediction we separate independent and target features. Here we will consider only **'Sex'**, **'Age'**, **'SibSp'**, **'Parch'**, **'Fare'**, **'Embarked'** only as the independent features and **Survived** as target variables because **PassengerId** will not affect the survival rate.

```
X = df3[['Pclass','Sex','Age', 'SibSp','Parch','Fare','Embarked']]
Y = df3['Survived']
```

### .
### Data formatting
This involves converting the data into a standard format or structure that can be easily processed by the algorithms or models used for analysis. 

Here we will discuss commonly used data formatting techniques i.e. Scaling and Normalization.

#### .
#### Scaling
- Scaling involves transforming the values of features to a specific range. It maintains the shape of the original distribution while changing the scale.
- Particularly useful when features have different scales, and certain algorithms are sensitive to the magnitude of the features.
- Common scaling methods include Min-Max scaling and Standardization (Z-score scaling).

**Min-Max Scaling:** Min-Max scaling rescales the values to a specified range, typically between 0 and 1. 

It preserves the original distribution and ensures that the minimum value maps to 0 and the maximum value maps to 1.

```
from sklearn.preprocessing import MinMaxScaler

# initialising the MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Numerical columns
num_col_ = [col for col in X.columns if X[col].dtype != 'object']
x1 = X

# learning the statistical parameters for each of the data and transforming
x1[num_col_] = scaler.fit_transform(x1[num_col_])
x1.head()
```
**Output:**
>  

**Standardization (Z-score scaling):** Standardization transforms the values to have a mean of 0 and a standard deviation of 1. It centers the data around the mean and scales it based on the standard deviation. 

Standardization makes the data more suitable for algorithms that assume a Gaussian distribution or require features to have zero mean and unit variance.

> Z = (X - μ) / σ

Where,

X = Data

μ = Mean value of X

σ = Standard deviation of X 


### .
### Data Cleansing Tools
- **OpenRefine:** A powerful open-source tool for cleaning and transforming messy data. It supports tasks like removing duplicate and data enrichment with easy-to-use interface.
- **Trifacta Wrangler:** A user-friendly tool designed for cleaning, transforming and preparing data for analysis. It uses AI to suggest transformations to streamline workflows.
- **TIBCO Clarity:** A tool that helps in profiling, standardizing and enriching data. It’s ideal to make high quality data and consistency across datasets.
- **Cloudingo:** A cloud-based tool focusing on de-duplication, data cleansing and record management to maintain accuracy of data.
- **IBM Infosphere Quality Stage:** It’s highly suitable for large-scale and complex data.

### .
### Adv and Disadv of Data Cleaning
#### Advantages:
- **Improved model performance:** Removal of errors, inconsistencies and irrelevant data helps the model to better learn from the data.
- **Increased accuracy:** Helps ensure that the data is accurate, consistent and free of errors.
- **Better representation of the data:** Data cleaning allows the data to be transformed into a format that better represents the underlying relationships and patterns in the data.
- **Improved data quality:** Improve the quality of the data, making it more reliable and accurate.
- **Improved data security:** Helps to identify and remove sensitive or confidential information that could compromise data security.

#### .
#### Disadvantages:
- **Time-consuming:** It is very time consuming task specially for large and complex datasets.
- **Error-prone:** It can result in loss of important information.
- **Cost and resource-intensive:** It is resource-intensive process that requires significant time, effort and expertise.

It can also require the use of specialized software tools.
Overfitting: Data cleaning can contribute to overfitting by removing too much data.

#### .
So we have discussed four different steps in data cleaning to make the data more reliable and to produce good results.

After properly completing the Data Cleaning steps, we’ll have a robust dataset that avoids any error and inconsistency. 

In summary, data cleaning is a crucial step in the data science pipeline that involves identifying and correcting errors, inconsistencies and inaccuracies in the data to improve its quality and usability.