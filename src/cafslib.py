# -*- coding: utf-8 -*-
"""
iCAFS Library

- train_model() - function used to train the model among a set of classifiers from scikit-learn.
- cafs()
- icafs()

January - 2025
-- Authors --
Salvador Romo
Miguel De-la-Torre
"""

#import cafslib as cafs
# Load required libraries and functions
import pandas as pd
import numpy as np
import random

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, auc

# Import the classifiers to be tested
from sklearn.naive_bayes import GaussianNB, BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Libraries for the ICAFS algorithm
from math import inf
from testflows.combinatorics import Covering


####################################################################
# Train the model. By now, there is a single model.
def train_model(model_name='GaussianNB'):
  """ To train the model
  """
  # Model parameters: Default parameters to avoid bias
  m_parameters = '()'
  if model_name == 'KNeighborsClassifier':
    m_parameters = '(n_neighbors=1)'
  
  m = eval(model_name + m_parameters)

  return m

# Original algorithm for covering array feature selection
def cafs(ca, dataset_x, dataset_y, max_iter, model='GaussianNB'):
  """
  CAFS - Stands for Covering Arrays Feature Selection.

  ca - The covering array
  dataset_x - the dataset "data"
  dataset_y - the dataset "labels"
  max_iter - the maximum number of iterations

  model - The classifier to be used in the evaluation
     'GaussianNB'
     'svm'
     'kNeighborsClassifier'
     'NeuralNetworks'
     'DecisionTreeClassifier'
  """

  # Required variables
  global_data_set = dataset_x
  global_max = 0
  max_iteartion = 0
  result_list_x = []
  result_list_y = []
  result_list_score = []
  num_rows = ca.shape[0]

  # Adjust the covering array according to the number of features in the dataset
  if len(dataset_x.columns) < ca.shape[1] :
    num_colums = len(dataset_x.columns)
  else:
    num_colums = ca.shape[1]

  # Iterate over the accuracy
  while max_iteartion < max_iter:

     lst_headers = global_data_set.columns.values.copy()
     max_score = 0.0
     mx_data_set = None
     #random.shuffle(lst_headers)

     for i in range(0,num_rows):
        lst_headers_to_select = []
        for j in range(0,num_colums ):
          if ca[i][j] == 1 :
              lst_headers_to_select.append(lst_headers[j])

        # with the list of headers to select get sub dat set of col with pandas
        if len(lst_headers_to_select) == 0:
            continue
        df_temp = dataset_x[lst_headers_to_select]
        x_train_temp, x_test_temp, y_train_temp, y_test_temp = train_test_split(dataset_x[lst_headers_to_select], dataset_y.values.ravel(),test_size=0.20,random_state=42)

        # train a model
        m = train_model( model )
        m.fit(x_train_temp, y_train_temp)
        y_pred = m.predict(x_test_temp)

        # get accuracy in terms of F1-score
        score = f1_score(y_test_temp, y_pred,average='macro')

        # if accuracy is better than maxScore so far assing subDataSet  to local_sub_data_set
        if score >= max_score :
            max_score = score
            mx_data_set = df_temp.copy()

     global_data_set = mx_data_set.copy()
     global_max = max_score
     print(m.get_params())
     print(global_data_set.columns)
     print(global_max)
     print("____________________________")
     num_colums  = len(global_data_set.columns)
     mx_data_set = None
     max_score = 0
     max_iteartion = max_iteartion + 1
     result_list_x.append(max_iteartion)
     result_list_y.append(len(global_data_set.columns))
     result_list_score.append(global_max)

  return result_list_x, result_list_y, result_list_score



#######################################################################################################
# TODO: Use model, the same as CAFS
# TODO: Test on the three datasets
######################################################################################################
def icafs(dataset_x, dataset_y, strenght, max_iter, model='GaussianNB'):
  # Required control variables
  max_iter_aux =  max_iter
  t_strenght = strenght
  v_variable = [0, 1]
  best_f1_score = float('-inf')
  best_data_set = dataset_x.columns.values.copy()
  max_iteration = 60
  max_len = inf
  score_list = []
  featur_list = []
  iter_list = []
  max_it = 0;

  # Iterate to reduce the number of features whenever it is possible.
  while (max_iter_aux >= 0):

      dict_parameters = {}
      partial_best_list = []
      partial_score = 0
      for colum_key in best_data_set:
          dict_parameters[colum_key] = v_variable
      random.shuffle(best_data_set)
      generate_covering_array = Covering(dict_parameters, strength=t_strenght)

      # Select a subset of features according to the generated covering array
      for test in generate_covering_array.array:
          list_attributes_to_consider = []

          check_for_all_cero = True
          for (test_key, test_value) in test.items():
              if test_value == 1:
                  check_for_all_cero = False
                  list_attributes_to_consider.append(test_key)

          if check_for_all_cero:
              continue
          df_temp = dataset_x.loc[ :,list_attributes_to_consider]

          # Organize data for train/test
          X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(df_temp, dataset_y.values.ravel(), test_size=0.20,
                                                                                  random_state=42)
          # Train the pre-defined model
          m = train_model( model )
          m.fit(X_train_temp, y_train_temp)

          y_pred = m.predict(X_test_temp)
          print(list_attributes_to_consider)
          score = f1_score(y_test_temp, y_pred,average='macro')
          #print(score)
          if score > partial_score :
              partial_score = score
              partial_best_list = list_attributes_to_consider
          m = None
      best_data_set = partial_best_list.copy()
      best_f1_score = partial_score
      print("best_f1_score= %s" % best_f1_score)
      print(best_data_set)
      print(len(best_data_set))
      print('---------------------')

      aux_data_score = best_f1_score
      score_list.append(aux_data_score)
      max_iter_aux = max_iter_aux-1
      max_it = max_it +1
      featur_list.append(len(best_data_set))
      iter_list.append(max_it)
  return score_list, featur_list, iter_list


