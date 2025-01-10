# -*- coding: utf-8 -*-
""" cis_cafs

Script to test the iterative covering array feature selection algorithm using the circle in square synthetic dataset

Salvador Romo and Miguel De-la-Torre
iteso and UdG working together.
"""
# from testflows.combinatorics import Covering

import random
import pandas as pd
import numpy as np
import math

# Our Library!!! - still under development
import cafslib as cl
from testflows.combinatorics import Covering

import time

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Circle

imgpath = '../images/' # TODO: Modify according to the test case and the classifier

############################## TEST CASE - CIRCLE IN SQUARE ################################

# The parameters of the covering array
paramter_to_test = {"1":[0,1],"2":[0,1],"3":[0,1],"4":[0,1],"5":[0,1],"6":[0,1]}
generate_covering_array = Covering(paramter_to_test, strength=3)

# print(generate_covering_array)

# Read/generate data
center = (0.5, 0.5)
radius = 0.5
circle = Circle(center, radius)

# fig, ax = plt.subplots()
# ax.add_patch(circle)
# ax.set_aspect('equal')

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
        # plt.plot(x,y,'ro')
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
        # plt.plot(x,y,'bo')
        points_outside+=1

#plt.savefig('/content/save_data/circle_in_the_sqaure.png', dpi=300, transparent=False, bbox_inches='tight')
# plt.show()

cisDf = pd.DataFrame(lst, columns=['X', 'Y','A','B','C','D','Result'])


X_cis = cisDf[['X', 'Y','A','B','C','D']]
y_cis= cisDf[['Result']]
#print(X_cis)

# Feature selection to obtain the selected features and scores
start_time = time.time()
res_scores, res_features, res_iter = cl.icafs(X_cis, y_cis, 3, 4)
end_time = time.time()
print(f"Processing time: {end_time - start_time} seconds")

# Convert to numpy arrays to operate over them
res_scores = np.array(res_scores)
res_features = np.array(res_features)
res_iter = np.array(res_iter)

# Generate figures
# 1. Features/F1-score and iterations
print('Generating figures...')
fig, ax1 = plt.subplots()
barwidth = 0.4
color = 'tab:red'
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Number of features', color=color)
ax1.set_title("CAFS Feature selection on the CIS dataset")
ax1.bar(res_iter-0.2, res_features, color=color, width=barwidth)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(1,max(res_iter)+2)
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
for i in range(len(res_iter)):
    ax1.text(i+1-0.2,    res_features[i], res_features[i])

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('F1_score', color=color)
ax2.bar(res_iter+0.2, res_scores, color=color, width=barwidth)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(min(res_scores)-0.001, max(res_scores)+0.001)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
for i in range(len(res_iter)):
    ax2.text(i+1, res_scores[i], f"{res_scores[i]:.3f}")

plt.savefig(imgpath + 'icafs_cis.png', bbox_inches='tight')
#plt.show()

# 2. Features vs F1-score
fig, ax1 = plt.subplots()
barwidth = 0.4

color = 'tab:blue'
ax1.set_xlabel('Number of features')
ax1.set_ylabel('F1-score', color=color)
ax1.set_title("CAFS Feature selection on the CIS dataset")
ax1.bar(np.arange(len(res_features))+1, res_scores, color=color, width=barwidth)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xticks(np.arange(len(res_features))+1, res_features.astype(str))
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

ax1.set_ylim(min(res_scores)-0.001, max(res_scores)+0.001)
for i in range(len(res_features)):
    ax2.text(i+1, res_scores[i], f"{res_scores[i]:.3f}")

plt.savefig(imgpath + 'icafs_cis_scores.png', bbox_inches='tight')
#plt.show()
