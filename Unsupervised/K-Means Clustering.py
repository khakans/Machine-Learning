# Import Library
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import random

data_df = pd.read_csv("kc_house_data.csv")
data_df.head()

# Memilih data yang akan di cluster
titik_a = "lat"
titik_b = "long"
dataset = data_df[[titik_a,titik_b]]
X = dataset
dataset.head()

plt.scatter(dataset[titik_a],dataset[titik_b],c='green')
plt.show()

# Fungsi menghitung nilai K random
def RandomCentroid(X):
  return X.sample(n=K)

# Membangkitkan nilai random untuk centroids awal
K=3
Centroids = RandomCentroid(X)

# Plot nilai random untuk centroids awal
plt.scatter(X[titik_a],X[titik_b],c='green', s=2)
plt.scatter(Centroids[titik_a],Centroids[titik_b],c='yellow',marker="*",s=100)
plt.show()

# Fungsi menghitung jarak
def EuclidianDistance(row_a, row_b):
  d1=(row_a[titik_a]-row_b[titik_a])**2
  d2=(row_a[titik_b]-row_b[titik_b])**2
  d=np.sqrt(d1+d2)
  return d

# Fungsi mencari nilai centroids baru
def PointClustering(pos,i,K,row,min_dist):
  for i in range(K):
    if row[i+1] < min_dist:
      min_dist = row[i+1]
      pos=i+1
  return pos  

# Fungsi menghitung perbedaan centroids
def diffCentroid(Centroids_new,Centroids):
  return (Centroids_new[titik_a] - Centroids[titik_a]).sum() + (Centroids_new[titik_b] - Centroids[titik_b]).sum()
  
# Fungsi Utama

miss = 1
j = 0

while(miss!=0):
    XD=X
    i=1
    for index1,row_a in Centroids.iterrows():
        ED=[]
        for index2,row_b in XD.iterrows():
            d = EuclidianDistance(row_a,row_b)
            ED.append(d)
        X[i]=ED
        i=i+1

    C=[]
    for index,row in X.iterrows():
        min_dist=row[1]
        pos=1
        pos = PointClustering(pos,i,K,row,min_dist)
        C.append(pos)
        
    X["Cluster"]=C
    Centroids_new = X.groupby(["Cluster"]).mean()[[titik_a,titik_b]]
    if j == 0:
        miss=1
        j=j+1
    else:
        miss = diffCentroid(Centroids_new,Centroids)
        miss.sum()
    Centroids = X.groupby(["Cluster"]).mean()[[titik_a,titik_b]]
    
# Plot hasil perhitungan
color=['green','yellow','red']
for k in range(K):
    dataFinalCluster=X[X["Cluster"]==k+1]
    plt.scatter(dataFinalCluster[titik_a],dataFinalCluster[titik_b],c=color[k], s=5)

plt.scatter(Centroids[titik_a],Centroids[titik_b],c='blue',marker="*",s=100)
plt.show()
