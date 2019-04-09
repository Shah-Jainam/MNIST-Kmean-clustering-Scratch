#!/usr/bin/env python
# coding: utf-8

# Importing Neccessory Libraries

# In[48]:


import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
import plotly.graph_objs as go
import plotly.offline as py
py.init_notebook_mode(connected=True)
get_ipython().run_line_magic('matplotlib', 'inline')


# Importing csv file and printing the head of t

# In[2]:


data = pd.read_csv('train.csv')


# In[3]:


data.head()


# Distribution of data

# In[4]:


num_stats = defaultdict(int)

for num in data['label']:
    num_stats[num] += 1

x = sorted(num_stats)
y = [num_stats[num] for num in x]

plt.bar(x, height=y)
plt.xlabel = 'Image Content'
plt.ylabel = 'Frequency'
plt.title = 'Distribution of MNIST'
plt.show()


# In[5]:


data.shape


# Droping the not applicables.

# In[6]:


data.dropna().shape


# removing the labels from dataset.

# In[7]:


result = data['label']
predictors = data.drop(['label'], axis=1)


# Standardising the values

# In[20]:


from sklearn.decomposition import PCA

X_std = StandardScaler().fit_transform(predictors.values)

pca = PCA(n_components=2)
x_2 = pca.fit(X_std).transform(X_std)


# K mean cluster from scratch
# defined init method and selected 9 clusters, tol is tolerance, max_iter is our cycle.
# Set a KMeans clustering with 9 components cuz there are 9 class labels
# 
# I have taken an empty dictionary which will be used for centroids and then I have used for loop I have assign a centroid. after start iterating using max_iter and I started with empty classification then calculate the distance of the features from our current centroid and classify them. 

# In[54]:


class K_Means:
    def __init__(self, k=9, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):

        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for featureset in data:
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
                    print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
                    optimized = False

            if optimized:
                break

    def predict(self, data):
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification


# After above process we will visualize our clusters using matplotlib

# In[58]:


kmeans = K_Means()
kmeans.fit(x_2)
kmeans.predict(x_2)

colors = 10*["green","red","gold","blue","silver", "yellow", "brown", "violet", "purple"]

for centroid in kmeans.centroids:
    plt.scatter(kmeans.centroids[centroid][0], kmeans.centroids[centroid][1],
                marker="o", color="k", s=150, linewidths=5)

for classification in kmeans.classifications:
    color = colors[classification]
    for featureset in kmeans.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker="x", color=color, s=150, linewidths=5)

plt.show()


# In[60]:


# with directly using Kmeans
# kmeans = KMeans(n_clusters=9)
# # Compute cluster centers and predict cluster indices
# kmeans_9 = kmeans.fit_predict(x_2)

# data = [
#     go.Scatter(
#         x= x_2[:, 0], 
#         y= x_2[:, 1], 
#         mode="markers",
#         showlegend=False,
#         marker=dict(
#             size=8,
#             color = kmeans_9,
#             colorscale = 'Rainbow',
#             showscale=False, 
#             line = dict(
#                 width = 2,
#                 color = 'rgb(255, 255, 255)'
#             )))]

# layout = go.Layout(
#     title= 'KMeans Clustering',
#     hovermode= 'closest',
#     xaxis= dict(
#          title= 'First Principal Component',
#         ticklen= 8,
#         zeroline= False,
#         gridwidth= 2,
#     ),
#     yaxis=dict(
#         title= 'Second Principal Component',
#         ticklen= 8,
#         gridwidth= 2,
#     ),
#     showlegend= True
# )

# fig = dict(data = data, layout = layout)
# py.iplot(fig, filename="kmeans_plot")

