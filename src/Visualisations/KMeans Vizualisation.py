import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from utils import KMeans0, spherical_kmeans_


data = make_moons(n_samples=1000,noise=10)
random_points=data[0]


Skmeans=KMeans0(k=6,dist_type='cos')
Slabels=Skmeans.fit(random_points)

kmeans=KMeans0(k=6,dist_type='euc')
labels=kmeans.fit(random_points)

s3 = spherical_kmeans_(random_points,n_clusters=6,block_size=50)

plt.figure(1,figsize=(15,8))
plt.subplot(2,3,1)
plt.scatter(random_points[:,0], random_points[:,1],c=labels)
plt.scatter(kmeans.centroids[:,0],kmeans.centroids[:,1],c=range(len(kmeans.centroids)),marker="*",s=200)
plt.axis('off')
plt.title('Kmeans')


plt.subplot(2,3,2)
plt.scatter(random_points[:,0], random_points[:,1])
plt.scatter(kmeans.centroids[:,0],kmeans.centroids[:,1],c=range(len(kmeans.centroids)),marker="*",s=200)
plt.axis('off')
plt.title('KMeans Centroids')

plt.subplot(2,3,4)
plt.scatter(random_points[:,0], random_points[:,1],c=s3[1])
plt.axis('off')
plt.title('Spherical Kmeans')



plt.subplot(2,3,5)
plt.scatter(random_points[:,0], random_points[:,1])
plt.scatter(s3[0][:,0],s3[0][:,1],c=range(len(s3[0])),marker="*",s=200)
plt.axis('off')
plt.title('Spherical Kmeans Centroids')

plt.subplot(2,3,6)
plt.scatter(random_points[:,0], random_points[:,1])
plt.scatter(s3[0][:,0],s3[0][:,1],c=range(len(s3[0])),marker="*",s=200)
plt.axis('off')
plt.xlim(-2,2)
plt.ylim(-2,2)
plt.title('Spherical Kmeans Centroids Zoomed')
plt.show()