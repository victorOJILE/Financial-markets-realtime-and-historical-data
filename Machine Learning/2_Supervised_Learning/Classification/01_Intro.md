## ML Classification 
This teaches a machine to sort things into categories.

It learns by looking at examples with labels (like emails marked "spam" or "not spam"). 

After learning, it can decide which category new items belong to, like identifying if a new email is spam or not. 
For example a classification model might be trained on dataset of images labeled as either dogs or cats and it can be used to predict the class of new and unseen images as dogs or cats based on their features such as color, texture and shape.

### .
### Types of Classification
`Binary Classification:`

In binary classification, the goal is to sort the data into two distinct categories. Think of it like a simple choice between two options. 

Imagine a system that sorts emails into either spam or not spam. It works by looking at different features of the email like certain keywords or sender details, and decides whether it’s spam or not. 

It only chooses between these two options.

#### .
`Multiclass Classification:`

Here, instead of just two categories, the data needs to be sorted into more than two categories. The model picks the one that best matches the input. 

Think of an image recognition system that sorts pictures of animals into categories like cat, dog, and bird.

Basically, machine looks at the features in the image (like shape, color, or texture) and chooses which animal the picture is most likely to be based on the training it received.

#### .
`Multi-Label Classification`

Unlike multiclass classification where each data point belongs to only one class, multi-label classification allows datapoints to belong to multiple classes.

A movie recommendation system could tag a movie as both action and comedy. The system checks various features (like movie plot, actors, or genre tags) and assigns multiple labels to a single piece of data, rather than just one.

Multilabel classification is relevant in specific use cases, but not as crucial for a starting overview of classification.

### .
### How Classification in ML Works?
Classification involves training a model using a labeled dataset, where each input is paired with its correct output label.

The model learns patterns and relationships in the data, so it can later predict labels for new, unseen inputs.

#### .
#### Here's how it works
`Data Collection:` You start with a dataset where each item is labeled with the correct class (for example, "cat" or "dog").

#### .
`Feature Extraction:` The system identifies features (like color, shape, or texture) that help distinguish one class from another. These features are what the model uses to make predictions.

#### .
`Model Training:` Classification - machine learning algorithm uses the labeled data to learn how to map the features to the correct class. It looks for patterns and relationships in the data.

#### .
`Model Evaluation:` Once the model is trained, it's tested on new, unseen data to check how accurately it can classify the items.

#### .
`Prediction:` After being trained and evaluated, the model can be used to predict the class of new data based on the features it has learned.

#### .
`Model Evaluation:` This helps us check how well the model performs and how good it is at handling new, unseen data. Depending on the problem and needs we can use different metrics to measure its performance.

---
If the quality metric is not satisfactory, the ML algorithm or hyperparameters can be adjusted, and the model is retrained. 

This iterative process continues until a satisfactory performance is achieved. 

In short, classification in machine learning is all about using existing labeled data to teach the model how to predict the class of new, unlabeled data based on the patterns it has learned.

### .
### ML Classification in Real Life
`Email spam filtering`

---
`Credit risk assessment`

Algorithms predict whether a loan applicant is likely to default by analyzing factors such as credit score, income, and loan history. 

This helps banks make informed lending decisions and minimize financial risk.

---
`Medical diagnosis`

Machine learning models classify whether a patient has a certain condition (e.g., cancer or diabetes) based on medical data such as test results, symptoms, and patient history. 

This aids doctors in making quicker, more accurate diagnoses, improving patient care.

---
`Image classification`

Applied in fields such as facial recognition, autonomous driving, and medical imaging.

---
`Sentiment analysis`

Determining whether the sentiment of a piece of text is positive, negative, or neutral. Businesses use this to understand customer opinions, helping to improve products and services.

---
`Fraud detection`

Algorithms detect fraudulent activities by analyzing transaction patterns and identifying anomalies crucial in protecting against credit card fraud and other financial crimes.

---
`Recommendation systems`

Used to recommend products or content based on past user behavior, such as suggesting movies on Netflix or products on Amazon. This personalization boosts user satisfaction and sales for businesses.

### .
### Classification Modeling in ML
`Class Separation`

Classification relies on distinguishing between distinct classes. The goal is to learn a model that can separate or categorize data points into predefined classes based on their features.

---
`Decision Boundaries`

The model draws decision boundaries in the feature space to differentiate between classes. These boundaries can be linear or non-linear.

---
`Sensitivity to Data Quality`

Classification models are sensitive to the quality and quantity of the training data. Well-labeled, representative data ensures better performance, while noisy or biased data can lead to poor predictions.

---
`Handling Imbalanced Data`

Classification problems may face challenges when one class is underrepresented. Special techniques like resampling or weighting are used to handle class imbalances.

---
`Interpretability`

Some classification algorithms, such as Decision Trees, offer higher interpretability, meaning it's easier to understand why a model made a particular prediction.

### .
### Classification Algorithms
Now, for implementation of any classification model it is essential to understand Logistic Regression, which is one of the most fundamental and widely used algorithms in machine learning for classification tasks. 

`Linear Classifiers:` Linear classifier models create a linear decision boundary between classes. They are simple and computationally efficient. Some of the linear classification models are as follows: 

- Logistic Regression
- Support Vector Machines having kernel = 'linear'
- Single-layer Perceptron
- Stochastic Gradient Descent (SGD) Classifier

#### .
`Non-linear Classifiers:` Non-linear models create a non-linear decision boundary between classes. They can capture more complex relationships between input features and target variable.

- K-Nearest Neighbours
- Kernel SVM
- Naive Bayes
- Decision Tree Classification

Ensemble learning classifiers: 
- Random Forests, 
- AdaBoost, 
- Bagging Classifier, 
- Voting Classifier, 

Extra Trees Classifier
- Multi-layer Artificial Neural Networks
