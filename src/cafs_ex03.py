# -*- coding: utf-8 -*-
"""CAFS_ex03

Salvador Romo and Miguel De-la-Torre
iteso and UdG working together.
"""

# Our Library!!! - still under development
import cafslib as cl

# Load required libraries and functions
import pandas as pd
import numpy as np
import random

from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import recall_score
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn import svm

from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import math

imgpath = '../images/'

# Load data
covering_array  = np.loadtxt("../data/coveringArray.csv", delimiter=",", dtype=int)

########################################## TEST CASE 1 - CACAO #####################################
df = pd.read_csv('../data/cacao.csv')
df_x = df.iloc[:, 1:]
df_y = df.iloc[:, 0:1]

scaler = MinMaxScaler()
df_norm_cacao = pd.DataFrame(scaler.fit_transform(df_x), columns=df_x.columns)

# Remove samples with NaN
df_norm_cacao_sinNaN = df_norm_cacao.dropna(how='any')
df_y_clean = df_y.loc[df_norm_cacao_sinNaN.index]
res_x,res_y,res_score = cl.cafs(covering_array, df_norm_cacao_sinNaN, df_y_clean, 10)

# Generate the figures
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Number of Features Selected for Cacao Nibs')
ax1.set_title("Features/Iteration")
ax1.bar(res_x, res_y, color = "red")

ax2.set_title("Scores/Iteration")
ax2.bar(res_x, res_score, color = "red")
ax2.set_ylim([0.5, 1.0])
plt.savefig(imgpath + 'cafs_cacao.png', bbox_inches='tight')


############################################ TEST 2 - CIRCLE IN SQUARE ################################
center = (0.5, 0.5)
radius = 0.5
circle = Circle(center, radius)

fig, ax = plt.subplots()
ax.add_patch(circle)
ax.set_aspect('equal')

lst = []
points_outside = 0
points_inside = 0

while(points_inside < 1000 ):
  x  = random.uniform(0, 1)
  y  = random.uniform(0, 1)
  a  = random.uniform(0, .1)
  b  = random.uniform(0, .2)
  c =  random.uniform(0, .3)
  d =  random.uniform(0, .4)

  if (x - center[0])**2 + (y - center[1])**2  <=  radius**2:
        lst.append([x,y,a,b,c,d,1])
        plt.plot(x,y,'ro')
        points_inside+=1

while(points_outside < 1000 ):
  x  = random.uniform(0, 1)
  y  = random.uniform(0, 1)
  a  = random.uniform(0, .1)
  b  = random.uniform(0, .2)
  c =  random.uniform(0, .3)
  d =  random.uniform(0, .4)

  if (x - center[0])**2 + (y - center[1])**2  >  radius**2:
        lst.append([x,y,a,b,c,d,0])
        plt.plot(x,y,'bo')
        points_outside+=1

plt.savefig(imgpath + 'circle_in_the_sqaure.png', dpi=300, transparent=False, bbox_inches='tight')
#plt.show()


df = pd.DataFrame(lst, columns=['X', 'Y','A','B','C','D','Result'])
df = df.sample(frac = 1)

X = df[['X', 'Y','A','B','C','D']]
y = df[['Result']]

CA2 =[[0, 0, 0, 1, 1, 1],[0, 0, 1, 0, 0, 0],[0, 1, 0, 0, 1, 0],[0 ,1 ,1 ,1 ,0 ,1 ],[1, 0, 0, 0, 0, 1],[1, 0, 1, 1, 1, 0],[1, 1 ,0 ,1 ,0 ,0],[1, 1, 1, 0, 1, 1],[0, 0, 0, 1, 0, 0
],[0, 0, 1, 0, 1, 1],[1, 1 ,0 ,1 ,1 ,1],[1, 1, 1, 0, 0, 0]]
ca_np = np.asarray(CA2, dtype=np.int32)

res_x_1,res_y_1,res_score_1 = cl.cafs(ca_np,X,y,2)

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Number of Features Selected for Circle In The Sqaure')
ax1.set_title("Features/Iteration")
ax1.bar(res_x_1, res_y_1, color = "red")

ax2.set_title("Scores/Iteration")
ax2.bar(res_x_1, res_score_1, color = "red")
plt.savefig(imgpath + 'cafs_cis.png', bbox_inches='tight')



"""# **Algarrobo Experiment**"""
"""
alagarrobo = pd.read_csv('/content/sample_data/algarrobo.csv')
algarrobo_x = alagarrobo.loc[:, 'R':'REDVI']
y_algarrobo = alagarrobo['Labels'].replace(to_replace=['N', 'P'], value=[0, 1])
y_algarrobo

X_algarrono = (algarrobo_x-algarrobo_x.min())/(algarrobo_x.max()-algarrobo_x.min())

algarrobo_x,algarrobo_y,algarrobo_score = cafs(covering_array,X_algarrono,y_algarrobo,10)
"""


"""**UMAP Vizualization**"""

"""
# Commented out IPython magic to ensure Python compatibility.
# %pip install umap-learn

import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import plotly.express as px

umap_2d = umap.UMAP(n_components=2)
umap_3d = umap.UMAP(n_components=3)

X_reduced_algarrobo = X_algarrono[['EXGR', 'MGRVI', 'RVI', 'DVI', 'EVI']];
X_reduced_algarrobo

X_reduced_standarized = StandardScaler().fit_transform(X_reduced_algarrobo)

algarrobo_umap_2d = umap_2d.fit_transform(X_reduced_standarized)
algarrobo_umap_3d = umap_3d.fit_transform(X_reduced_standarized)

plt.scatter(
    algarrobo_umap_2d[:, 0],
    algarrobo_umap_2d[:, 1],
    c=[sns.color_palette()[x] for x in alagarrobo.Labels.map({"N":0, "P":1})])

plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP Algarrobo', fontsize=24);

fig_3d = px.scatter_3d(
    algarrobo_umap_3d, x=0, y=1, z=2,
    color=alagarrobo.Labels, labels={'color': 'species'}
)
fig_3d.update_traces(marker_size=5)
fig_3d.show()

fig = plt.figure()

# syntax for 3-D projection
ax = plt.axes(projection ='3d')

# plotting
ax.scatter(algarrobo_umap_3d[:, 0], algarrobo_umap_3d[:, 1], algarrobo_umap_3d[:, 2], c = colors)
plt.show();
"""




"""# **Cacao Experiment**"""
"""
cacao_reduced = df_norm_cacao[['1225', '1322', '1559', '1936', '2296']]
cacao_reduced

X_reduced_cacao_standarized = StandardScaler().fit_transform(cacao_reduced)

umap_2d_cacao = umap.UMAP(n_components=2)
umap_3d_cacao = umap.UMAP(n_components=3)

cacao_umap_2d = umap_2d_cacao.fit_transform(X_reduced_cacao_standarized)
cacao_umap_3d = umap_3d_cacao.fit_transform(X_reduced_cacao_standarized)

fig_3d = px.scatter_3d(
    cacao_umap_3d, x=0, y=1, z=2,
    color=df['10101010.00'], labels={'color': 'species'}
)
fig_3d.update_traces(marker_size=5)
fig_3d.show()

from sklearn.decomposition import PCA

pca_2d_cacao = PCA(n_components=2)
pca_3d_cacao = PCA(n_components=3)

cacao_2d_pca = pca_2d_cacao.fit_transform(X_reduced_cacao_standarized)
cacao_3d_pca = pca_3d_cacao.fit_transform(X_reduced_cacao_standarized)

cacao_2d_pca

fig_3d = px.scatter_3d(
    cacao_3d_pca, x=0, y=1, z=2,
    color=df['10101010.00'], labels={'color': 'species'}
)
fig_3d.update_traces(marker_size=5)
fig_3d.show()

df = df.rename(columns={'10101010.00': 'variaty'})
plt.scatter(
    cacao_2d_pca[:, 0],
    cacao_2d_pca[:, 1],
    c=[sns.color_palette()[x] for x in df.variaty.map({ 1 : 0, 2 :1 , 3 :2, 4 :3, 5 :4, 6 :5 })])

plt.gca().set_aspect('equal', 'datalim')
plt.title('PCA Algarrobo', fontsize=24);
"""

