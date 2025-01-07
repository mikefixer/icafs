# Script to test the iterative covering array feature selection algorithm using the circle in square synthetic dataset
#
from testflows.combinatorics import Covering

import random
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import pandas as pd

import numpy as np
import math

import cafslib as cl


paramter_to_test = {"1":[0,1],"2":[0,1],"3":[0,1],"4":[0,1],"5":[0,1],"6":[0,1]}
generate_covering_array = Covering(paramter_to_test, strength=3)

print(generate_covering_array)

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

#plt.savefig('/content/save_data/circle_in_the_sqaure.png', dpi=300, transparent=False, bbox_inches='tight')
# plt.show()


cisDf = pd.DataFrame(lst, columns=['X', 'Y','A','B','C','D','Result'])


X_cis = cisDf[['X', 'Y','A','B','C','D']]
y_cis= cisDf[['Result']]
print(X_cis)


score_cis,festure_cis,iter_cis = cl.icafs(X_cis, y_cis, 3, 4, model='GaussianNB')

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.set_title("Features/Iteration")
ax1.bar(iter_cis, festure_cis, color = "red")

ax2.set_title("Scores/Iteration")
ax2.bar(iter_cis, score_cis, color = "red")
#plt.savefig('/content/saveimage/icafs_cis.png', bbox_inches='tight')
plt.show()

