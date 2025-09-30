# Hierarchical Clustering
This is an unsupervised learning method for clustering data points. The algorithm builds clusters by measuring the dissimilarities between data.

Unsupervised learning means that a model does not have to be trained, and we do not need a "target" variable. 

This method can be used on any data to visualize and interpret the relationship between individual data points.

Here we will use hierarchical clustering to group data points and visualize the clusters using both a dendrogram and scatter plot.

## .
### How does it work?
We will use Agglomerative Clustering, a type of hierarchical clustering that follows a bottom up approach. 

We begin by treating each data point as its own cluster. Then, we join clusters together that have the shortest distance between them to create larger clusters.

This step is repeated until one large cluster is formed containing all of the data points.

#### .
Hierarchical clustering requires us to decide on both a distance and linkage method. We will use euclidean distance and the Ward linkage method, which attempts to minimize the variance between clusters.

#### Example
Start by visualizing some data points:

```
import numpy as np
import matplotlib.pyplot as plt

# Create arrays that resemble two variables in a dataset. 
# Note that while we only use two variables here, 
# this method will work with any number of variables

x = [4, 5, 10, 4, 3, 11, 14 , 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

# plt.scatter(x, y)
# plt.show()

# Now we compute the ward linkage using euclidean distance, 
# and visualize it using a dendrogram

from scipy.cluster.hierarchy import dendrogram, linkage

# Turn the data into a set of points:
data = list(zip(x, y))
print(data)

# [(4, 21), (5, 19), (10, 24), 
#   (4, 17), (3, 16), (11, 25), 
#   (14, 24), (6, 22), (10, 21), (12, 21)]

# Compute the linkage between all of the different points. 
# Here we use a simple euclidean distance measure and Ward's linkage

linkage_data = linkage(data, method='ward', metric='euclidean')
dendrogram(linkage_data)

# plt.show()

# Here, we do the same thing with Python's scikit-learn library. 
# Then, visualize on a 2-dimensional plot

from sklearn.cluster import AgglomerativeClustering

hierarchical_cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
labels = hierarchical_cluster.fit_predict(data)
print(labels)

# [0 0 1 0 0 1 1 0 1 1]

plt.scatter(x, y, c=labels)
plt.show()
```