# Data preprocessing
This is an important step in the data science transforming raw data into a clean structured format for analysis. 

It involves tasks like handling missing values, normalizing data and encoding variables. 

Mastering preprocessing in Python ensures reliable insights for accurate predictions and effective decision-making.

> `Raw data` ==> 
`Structure data` ==> 
`Data Preprocessing` ==> 
`Exploration Data Analysis (EDA)` ==>
`Insights, Reports, Visual graphs`

## .
## Steps in Data Preprocessing
### Import the necessary libraries

```
# importing libraries

import pandas as pd
import scipy
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
```

### .
### Load the dataset
You can download dataset from https://media.geeksforgeeks.org/wp-content/ uploads/20250115110111213229/diabetes.csv.

```
df = pd.read_csv('Geeksforgeeks/Data/diabetes.csv')
print(df.head())
```

---
**Output:**

| Preg nancies | Glu cose | BP | Skin Thickness | Insulin | BMI |
|---|---|---|---|---|---|
|0|6|148|72|35|0|

### .
### Check the data info
```df.info()```

If there is Null values in the dataset, we can also check them using df.isnull()

```
df.isnull().sum()
```

### .
### Statistical Analysis
In statistical analysis we use df.describe() which will give a descriptive overview of the dataset.

```
df.describe()
```
**Output:**
>  

The above table shows the count, mean, standard deviation, min, 25%, 50%, 75% and max values for each column. 

When we carefully observe the table we will find that Insulin, Pregnancies, BMI, BloodPressure columns has outliers. 

Let's plot the boxplot for each column for easy understanding.

### .
### Check the outliers
```
# Box Plots
fig, axs = plt.subplots(9,1,dpi=95, figsize=(7,17))
i = 0

for col in df.columns:
    axs[i].boxplot(df[col], vert=False)
    axs[i].set_ylabel(col)
    i+=1

plt.show()
```
**Output:**
>   

From the above boxplot we can clearly see that every column has some amounts of outliers. 

### .
### Drop the outliers

```
# Identify the quartiles
q1, q3 = np.percentile(df['Insulin'], [25, 75])

# Calculate the interquartile range
iqr = q3 - q1

# Calculate the lower and upper bounds
lower_bound = q1 - (1.5 * iqr)
upper_bound = q3 + (1.5 * iqr)

# Drop the outliers
clean_data = df[(df['Insulin'] >= lower_bound) 
   & (df['Insulin'] <= upper_bound)]

# Do the same for other columns head
# Pregnancies, Age and so on ...
```

### .
### Correlation
```
#correlation
corr = df.corr()

plt.figure(dpi=130)
sns.heatmap(df.corr(), annot=True, fmt= '.2f')

plt.show()
```
**Output:**
>    

We can also compare by single columns in descending order
```
corr['Outcome'].sort_values(ascending = False)
```
**Output:**
>   

### .
### Check Outcomes Proportionality
```
plt.pie(df.Outcome.value_counts(), 
  labels= ['Diabetes', 'Not Diabetes'], 
  autopct='%.f', shadow=True)
plt.title('Outcome Proportionality')

plt.show()
```
**Output:**

### .
### Separate independent features and Target Variables

```
# separate array into input and output components
X = df.drop(columns =['Outcome'])
Y = df.Outcome
```

### .
### Normalization or Standardization
#### Normalization
Normalization works well when the features have different scales and the algorithm being used is sensitive to the scale of the features, such as k-nearest neighbors or neural networks.

Rescale your data using scikit-learn using the MinMaxScaler.
MinMaxScaler scales the data so that each feature is in the range [0, 1]. 

```
# initialising the MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# learning the statistical parameters for each of the data and transforming
rescaledX = scaler.fit_transform(X)
rescaledX[:5]
```
#### Output:
>  

#### .
#### Standardization
Standardization is a useful technique to transform attributes with a Gaussian distribution and differing means and standard deviations to a standard Gaussian distribution with a mean of 0 and a standard deviation of 1.

We can standardize data using scikit-learn with the StandardScaler class.

It works well when the features have a normal distribution or when the algorithm being used is not sensitive to the scale of the features

```
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
rescaledX[:5]
```
#### Output:
>  

#### .
In conclusion data preprocessing is an important step to make raw data clean for analysis. 

Using Python we can handle missing values, organize data and prepare it for accurate results. This ensures our model is reliable and helps us uncover valuable insights from data.