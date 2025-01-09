# -*- coding: utf-8 -*-
""" cis_cafs

Experimentación 

Salvador Romo and Miguel De-la-Torre
iteso and UdG working together.
"""

# Our Library!!! - still under development
import cafslib as cl

# Load required libraries and functions
import pandas as pd
import numpy as np
import random

# from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import StandardScaler
#from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
#from sklearn.metrics import accuracy_score
#from sklearn.metrics import recall_score

from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import math

import time

imgpath = '../images/'

# Load data
covering_array  = np.loadtxt("../data/coveringArray.csv", delimiter=",", dtype=int)

############################## TEST CASE - CIRCLE IN SQUARE ################################

# Read/generate data
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

plt.savefig(imgpath + 'cis_cafs.png', dpi=300, transparent=False, bbox_inches='tight')
#plt.show()

# Organize data and proceed with pre-processing
df = pd.DataFrame(lst, columns=['X', 'Y','A','B','C','D','Result'])
df = df.sample(frac = 1)

X = df[['X', 'Y','A','B','C','D']]
y = df[['Result']]

# Read/generate the Covering Array
CA2 =[[0, 0, 0, 1, 1, 1],[0, 0, 1, 0, 0, 0],[0, 1, 0, 0, 1, 0],[0 ,1 ,1 ,1 ,0 ,1 ],[1, 0, 0, 0, 0, 1],[1, 0, 1, 1, 1, 0],[1, 1 ,0 ,1 ,0 ,0],[1, 1, 1, 0, 1, 1],[0, 0, 0, 1, 0, 0
],[0, 0, 1, 0, 1, 1],[1, 1 ,0 ,1 ,1 ,1],[1, 1, 1, 0, 0, 0]]
ca_np = np.asarray(CA2, dtype=np.int32)

# Feature selection to obtain the scores
start_time = time.time()
res_x_1,res_y_1,res_score_1 = cl.cafs(ca_np,X,y,2)
end_time = time.time()
print(f"Processing time: {end_time - start_time} seconds")

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Number of Features Selected for Circle In The Sqaure')
ax1.set_title("Features/Iteration")
ax1.bar(res_x_1, res_y_1, color = "red")

ax2.set_title("Scores/Iteration")
ax2.bar(res_x_1, res_score_1, color = "red")
plt.savefig(imgpath + 'cafs_cis.png', bbox_inches='tight')

