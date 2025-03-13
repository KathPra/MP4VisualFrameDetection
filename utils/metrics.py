from math import log
import numpy as np
import math
from sklearn.metrics.cluster import contingency_matrix
from collections import Counter
from statistics import mean

def variation_of_information(X, Y):
  """
  X,Y are lists of list containing node ids
  """
  n = float(sum([len(x) for x in X]))
  sigma = 0.0
  for x in X:
    p = len(x) / n
    for y in Y:
      q = len(y) / n
      r = len(set(x) & set(y)) / n
      if r > 0.0:
        sigma += r * (log(r / p, 2) + log(r / q, 2))
  return abs(sigma)


def entropy(labels, base=None):
  # compute label probabilities
  _,counts = np.unique(labels, return_counts=True)
  norm_counts = counts / counts.sum()
  print("Number of classes", len(norm_counts))
  # compute entropy
  base = math.e if base is None else base
  return -(norm_counts * np.log(norm_counts)/np.log(base)).sum()


def conditional_entropy(clust_a, clust_b):
    """
    Input:
    1) clust_a: list of cluster labels
    2) clust_b: list of cluster labels

    Returns the conditional entropy H(A|B), H(B|A)
    """
    # Compute the contingency matrix (also known as confusion matrix)
    contingency = contingency_matrix(clust_a, clust_b)
    
    # Normalize the contingency matrix to get joint probabilities
    joint_prob = contingency / np.sum(contingency)
    
    # Compute the marginal probabilities
    marginal_prob_a = np.sum(joint_prob, axis=1)
    marginal_prob_b = np.sum(joint_prob, axis=0)
    
    # Compute the conditional entropy H(A|B)
    conditional_entropy_ab = -np.sum(joint_prob * np.log(joint_prob / (marginal_prob_a[:, None] + 1e-10)))
    conditional_entropy_ba = -np.sum(joint_prob * np.log(joint_prob / (marginal_prob_b[None, :] + 1e-10)))

    return conditional_entropy_ab, conditional_entropy_ba


def cond_entropy(label, label_cond):
    """
    Input:
    1) label: list of labels
    2) label_cond: list of labels

    Returns the conditional entropy H(label|label_cond), H(label_cond|label) and VI
    """
    # Compute the contingency matrix (also known as confusion matrix)
    contingency = contingency_matrix(label, label_cond)
    # Normalize the contingency matrix to get joint probabilities
    joint_prob = contingency / np.sum(contingency)
    # compute conditional entropy
     # Compute the marginal probabilities
    marginal_prob_a = np.sum(joint_prob, axis=1)
    marginal_prob_b = np.sum(joint_prob, axis=0)
    
    # Compute the conditional entropy H(A|B)
    eps = 1e-10

    cond_prob_ab = joint_prob / (marginal_prob_a[:, None] + eps)
    cond_prob_ba = joint_prob / (marginal_prob_b[None, :] + eps)
    conditional_entropy_ab = -np.sum(joint_prob * np.log(cond_prob_ab + eps))
    conditional_entropy_ba = -np.sum(joint_prob * np.log(cond_prob_ba + eps))
    VI = conditional_entropy_ab + conditional_entropy_ba
    delta = conditional_entropy_ab - conditional_entropy_ba
    print(conditional_entropy_ab, "vs.", conditional_entropy_ba, delta)

    return conditional_entropy_ab,conditional_entropy_ba, VI


def cluster_quali(clust_overview, class_mapping):
    """"
    determine how many clusters contain samples from more than one class

    Input:
    1) clust_overview: dictionary with cluster ids as keys and list of image ids as values
    2) class_mapping: dictionary with image ids as keys and class labels as values

    Prints:
    1) the mean cluster size
    2) the median cluster size
    3) the percentage of clusters with size 1
    4) the number of clean clusters
    5) the percentage of data points in clean clusters
    6) the classes that are never confused with other classes
    """    

    print("the mean cluster size is: ", mean([len(clust_overview[c]) for c in clust_overview]))
    print("the median cluster size is: ", np.median([len(clust_overview[c]) for c in clust_overview]))
    print(" datapoints are in clusters of size 1:", sum([len(clust_overview[c]) for c in clust_overview if len(clust_overview[c]) == 1]) / sum([len(clust_overview[c]) for c in clust_overview]) * 100 , "%")

    clean_clust = 0
    mixed_clust = []
    for c in clust_overview:
        classes = [class_mapping[img] for img in clust_overview[c]]
        class_counter = Counter(classes)
        classes_set = set(classes)
        if len(classes_set) > 1:
            print("Cluster ", c, "contains", len(clust_overview[c]),"samples from ", len(classes_set), " classes, i.e. ,", classes_set,".", class_counter)
            for cl in class_counter:
                if class_counter[cl] < 5:
                    print("Examples images from cluster ", c, "belonging to class ", cl, ":", [img for img in clust_overview[c] if class_mapping[img] == cl])
            mixed_clust.append(c for c in classes_set)
        else: clean_clust += 1
    print("Number of clean clusters: ", clean_clust,"these are", np.round(clean_clust/len(clust_overview)*100,2), "%")
    print("Number of data points in clean clusters: ", sum([len(clust_overview[c]) for c in clust_overview if len(set([class_mapping[img] for img in clust_overview[c]])) == 1]) / sum([len(clust_overview[c]) for c in clust_overview]) * 100 , "%")
    never_confused = [x for x in classes if x not in mixed_clust]
    if len(never_confused): print("Class(es)", never_confused, "are never confused with other classes.")

def cluster_quali_unsup(clust_overview):
    """"
    determine how many clusters contain samples from more than one class

    Input:
    1) clust_overview: dictionary with cluster ids as keys and list of image ids as values

    Prints:
    1) the mean cluster size
    2) the median cluster size
    3) the max cluster size
    4) the min cluster size
    5) the percentage of clusters with size 1
    6) the largest clusters (top 10)
    """    

    print("the mean cluster size is: ", mean([len(clust_overview[c]) for c in clust_overview]))
    print("the median cluster size is: ", np.median([len(clust_overview[c]) for c in clust_overview]))
    print("the max cluster size is: ", np.max([len(clust_overview[c]) for c in clust_overview]))
    print("the min cluster size is: ", np.min([len(clust_overview[c]) for c in clust_overview]))
    print(" datapoints are in clusters of size 1:", sum([len(clust_overview[c]) for c in clust_overview if len(clust_overview[c]) == 1]) / sum([len(clust_overview[c]) for c in clust_overview]) * 100 , "%")
    print(" the largest clusters have sizes: ", sorted([len(clust_overview[c]) for c in clust_overview], reverse=True)[:10])
