
from matplotlib import pyplot as plt
import numpy as np






from sklearn.metrics import roc_curve
import random

from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
import warnings
import pickle
warnings.filterwarnings("ignore")

# from src.data_assemble.assemble_ml import *

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


def metrics(model, X_test, y_test, plot_roc=False):
  # y_pred = model.predict(X_test)
  lr_probs = model.predict_proba(X_test)
  
  # keep probabilities for the positive outcome only
  lr_probs = lr_probs[:, 1]
  # calculate scores
  ns_probs = [1 for _ in range(len(y_test))]
  ns_auc = roc_auc_score(y_test, ns_probs)
  lr_auc = roc_auc_score(y_test, lr_probs)
  # summarize scores
  print('No Skill: ROC AUC=%.3f' % (ns_auc))
  print('Clf: ROC AUC=%.3f' % (lr_auc))
  # calculate roc curves
  ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
  lr_fpr, lr_tpr, thresholds = roc_curve(y_test, lr_probs)
  # plot the roc curve for the model
  gmeans = np.sqrt(lr_tpr * (1-lr_fpr))
  ix = np.argmax(gmeans)
  roc_auc = lr_auc
  if plot_roc:
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='Constant 1')
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Clf')
    plt.scatter(lr_fpr[ix], lr_tpr[ix], marker='o', color='black', label='Best')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()
    
  
    print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
  y_pred = lr_probs >= 0.5#thresholds[ix]
  acc = accuracy_score(y_test, y_pred)
  precision = precision_score(y_test, y_pred)
  recall = recall_score(y_test, y_pred)
  f1 = f1_score(y_test, y_pred)
  print(confusion_matrix(y_test, y_pred))
  metrics = [acc, precision, recall, f1, roc_auc]
  return metrics

def metrics_wnd(model, X_test, y_test, wnd, plot_roc=False):
  # y_pred = model.predict(X_test)
  lr_probs = model.predict_proba(X_test)
  # keep probabilities for the positive outcome only
  lr_probs = lr_probs[:, 1]
  if len(lr_probs) < wnd:
    lr_probs_new = [lr_probs.max() for _ in range(len(lr_probs))]
  else: 
    lr_probs_new = []
    for i in range(wnd):
        lr_probs_new.append(lr_probs[:wnd].max())
    for i in range(wnd, len(lr_probs) - wnd):
        lr_probs_new.append(lr_probs[i - wnd:i + wnd].max())
    for i in range(len(lr_probs) - wnd, len(lr_probs)):
        lr_probs_new.append(lr_probs[-wnd:].max())
  # calculate scores
  ns_probs = [1 for _ in range(len(y_test))]
  ns_auc = roc_auc_score(y_test, ns_probs)
  lr_auc = roc_auc_score(y_test, lr_probs_new)
  # summarize scores
  print('No Skill: ROC AUC=%.3f' % (ns_auc))
  print('Clf: ROC AUC=%.3f' % (lr_auc))
  # calculate roc curves
  ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
  lr_fpr, lr_tpr, thresholds = roc_curve(y_test, lr_probs_new)
  # plot the roc curve for the model
  gmeans = np.sqrt(lr_tpr * (1-lr_fpr))
  ix = np.argmax(gmeans)
  roc_auc = lr_auc
  if plot_roc:
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='Constant 1')
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Clf')
    plt.scatter(lr_fpr[ix], lr_tpr[ix], marker='o', color='black', label='Best')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()
    
  
    print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
  y_pred = [lr_probs_new[i] >= 0.5 for i in range(len(lr_probs_new))]#thresholds[ix]
  acc = accuracy_score(y_test, y_pred)
  precision = precision_score(y_test, y_pred)
  recall = recall_score(y_test, y_pred)
  f1 = f1_score(y_test, y_pred)
  print(confusion_matrix(y_test, y_pred))
  metrics = [acc, precision, recall, f1, roc_auc]
  return metrics

def train_test_splitter(list_to_split, split_ratio=0.4):
    random.shuffle(list_to_split)
    elements = len(list_to_split)
    middle = int(elements * split_ratio)
    return [list_to_split[:middle], list_to_split[middle:]]

def get_split_train_test(dataset_stations, split_ratio=0.4):
    splitted_test, splitted_train = train_test_splitter(dataset_stations, split_ratio=split_ratio)
    try:
        X_train = pd.concat([v[0] for v in splitted_train], axis=0)
    except TypeError:
        X_train = np.concatenate([v[0] for v in splitted_train], axis=0)
    y_train = np.concatenate([v[1] for v in splitted_train])

    try:
        X_test = pd.concat([v[0] for v in splitted_test], axis=0)
    except TypeError:
        X_test = np.concatenate([v[0] for v in splitted_test], axis=0)
    y_test = np.concatenate([v[1] for v in splitted_test])

    return X_train, y_train, X_test, y_test

def train_test_clf(clf, dataset_stations, split_ratio=0.4):
    X_train, y_train, X_test, y_test = get_split_train_test(dataset_stations, split_ratio)
    # try:
    #     print("incrementing")
        
    #     clf.fit(X_train, y_train, xgb_model=clf.get_booster())
    # except:
        
    clf.fit(X_train, y_train)
    train_metrics = metrics(clf, X_train, y_train)
    test_metrics = metrics(clf, X_test, y_test)
    print('On train:', train_metrics)
    print('On test:', test_metrics)
    return clf, test_metrics
