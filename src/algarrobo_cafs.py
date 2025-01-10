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

from sklearn.preprocessing import MinMaxScaler, StandardScaler

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import time

imgpath = '../images/'

############################## TEST CASE - ALGARROBO #####################################

# **Algarrobo Experiment**
dfAlagarrobo = pd.read_csv('../data/algarrobo.csv')
X = dfAlagarrobo.loc[:, 'R':'REDVI']
y = dfAlagarrobo['Labels'].replace(to_replace=['N', 'P'], value=[0, 1])

# Normalize data (some samples fall outside the feature space, due to additive noise)
scaler = MinMaxScaler()
X_norm = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Remove samples with NaN
X_norm_sNaN = X_norm.dropna(how='any')
y_sNaN = y.loc[X_norm_sNaN.index]

# Feature selection to obtain the scores
start_time = time.time()
res_iter, res_features, res_scores = cl.cafs(X_norm_sNaN, y_sNaN, 5)
end_time = time.time()
print(f"Processing time: {end_time - start_time} seconds")

########################### DISPLAY RESULTS ###############################
# Generate figures
# Convert to numpy arrays to operate over them
res_scores = np.array(res_scores)
res_features = np.array(res_features)
res_iter = np.array(res_iter)

# 1. Features/F1-score and iterations
print('Generating figures...')
fig, ax1 = plt.subplots()
barwidth = 0.4
color = 'tab:red'
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Number of features', color=color)
ax1.set_title("CAFS Feature selection on the Algarrobo dataset")
ax1.bar(res_iter-0.2, res_features, color=color, width=barwidth)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(1,max(res_features)+2)
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

plt.savefig(imgpath + 'cafs_algarrobo.png', bbox_inches='tight')
#plt.show()

# 2. Features vs F1-score
fig, ax1 = plt.subplots()
barwidth = 0.4

color = 'tab:blue'
ax1.set_xlabel('Number of features')
ax1.set_ylabel('F1-score', color=color)
ax1.set_title("CAFS Feature selection on the Algarrobo dataset")
ax1.bar(np.arange(len(res_features))+1, res_scores, color=color, width=barwidth)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xticks(np.arange(len(res_features))+1, res_features.astype(str))
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

ax1.set_ylim(min(res_scores)-0.001, max(res_scores)+0.001)
for i in range(len(res_features)):
    ax2.text(i+1, res_scores[i], f"{res_scores[i]:.3f}")

plt.savefig(imgpath + 'cafs_algarrobo_scores.png', bbox_inches='tight')
#plt.show()
