# -*- coding: utf-8 -*-
""" cis_cafs

Experimentación 

Salvador Romo and Miguel De-la-Torre
iteso and UdG working together.
"""

# The Covering Array Feature Selection Library - cafs and icafs
# import cafslib as cl

# Load required libraries and functions
import pandas as pd
import numpy as np
import random

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import f1_score
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import recall_score

from matplotlib.patches import Circle
import matplotlib.pyplot as plt
# from matplotlib.ticker import MaxNLocator
import math

imgpath = '../images/'

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

while(points_inside < 5000 ):
  x  = random.uniform(0, 1)
  y  = random.uniform(0, 1)
  ax  = random.uniform(0, 0.1) # Additive noise 10%
  ay  = random.uniform(0, 0.1)
  bx  = random.uniform(0, 0.2) # Additive noise 20%
  by  = random.uniform(0, 0.2)
  cx =  random.uniform(0, 0.3) # Additive noise 30%
  cy =  random.uniform(0, 0.3)
  dx =  random.uniform(0, 0.4) # Additive noise 40%
  dy =  random.uniform(0, 0.4)

  if (x - center[0])**2 + (y - center[1])**2  <=  radius**2:
        lst.append([x, y, x+ax, y+ay, x+bx, y+by, x+cx, y+cy, x+dx, y+dy, 1])
        plt.plot(x,y,'ro')
        points_inside+=1

while(points_outside < 5000 ):
  x  = random.uniform(0, 1)
  y  = random.uniform(0, 1)
  ax  = random.uniform(0, 0.1) # Additive noise 10%
  ay  = random.uniform(0, 0.1)
  bx  = random.uniform(0, 0.2) # Additive noise 20%
  by  = random.uniform(0, 0.2)
  cx =  random.uniform(0, 0.3) # Additive noise 30%
  cy =  random.uniform(0, 0.3)
  dx =  random.uniform(0, 0.4) # Additive noise 40%

  if (x - center[0])**2 + (y - center[1])**2  >  radius**2:
        lst.append([x, y, x+ax, y+ay, x+bx, y+by, x+cx, y+cy, x+dx, y+dy, 0])
        plt.plot(x,y,'bo')
        points_outside+=1

plt.savefig(imgpath + 'cis.png', dpi=300, transparent=False, bbox_inches='tight')
plt.show()

# Organize data and proceed with pre-processing
df = pd.DataFrame(lst, columns=['X', 'Y', 'X10', 'Y10', 'X20', 'Y20', 'X30', 'Y30', 'X40', 'Y40', 'Result'])
df = df.sample(frac = 1)

######################################################################################
# Save the dataset (fixed for comparison purposes)
# Generated on January 10, 2025: uncomment to re-generate
# df.to_csv('../data/cis_data.csv', index=False)

# INFO: Reading the dataset. Separate the dataframes for features and labels.
# dfCIS = pd.read_csv('../data/cis_data.csv')
# X = dfCIS[['X', 'Y', 'X10', 'Y10', 'X20', 'Y20', 'X30', 'Y30', 'X40', 'Y40']]
# y = dfCIS[['Result']]
