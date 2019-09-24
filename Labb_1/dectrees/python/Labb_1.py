#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 14:27:43 2019

@author: jonteyh
"""
import PyQt5
import monkdata as dataset
import dtree
import dtree as tree
import numpy as np
import pandas as pa
import random
from matplotlib import pyplot as pt


"Assignment-1"
print('The Entropy for MONK-1 '+str(dtree.entropy(dataset.monk1)))
print('The Entropy for MONK-2 '+str(dtree.entropy(dataset.monk2)))
print('The Entropy for MONK-3 '+str(dtree.entropy(dataset.monk3)))

"Assignment-3"
print('Monk1-data')
print('\n')
for x in range(6):
    print('a-number',x+1,'Average-entropy',dtree.averageGain(dataset.monk1,dataset.attributes[x]))
    print('\n')
    

print('Monk2-data')
print('\n')
for x in range(6):
    print('a-number',x+1,'Average-entropy',dtree.averageGain(dataset.monk2,dataset.attributes[x]))
    print('\n')
    

print('Monk3-data')
print('\n')    
for x in range(6):
    print('a-number',x+1,'Average-entropy',dtree.averageGain(dataset.monk3,dataset.attributes[x]))
    print('\n')
    

"Assignment-5"
t_1 = tree.buildTree(dataset.monk1,dataset.attributes)
print('Monk1-Train',1-tree.check(t_1,dataset.monk1))
print('Monk1-Test',1-tree.check(t_1,dataset.monk1test))
print('\n')

t_2 = tree.buildTree(dataset.monk2,dataset.attributes)
print('Monk2-Train',1-tree.check(t_2,dataset.monk2))
print('Monk2-Test',1-tree.check(t_2,dataset.monk2test))
print('\n')

t_3 = tree.buildTree(dataset.monk3,dataset.attributes)
print('Monk3-Train',1-tree.check(t_3,dataset.monk3))
print('Monk3-Test',1-tree.check(t_3,dataset.monk3test))
print('\n')

"Assignment-7"
def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata))
    return ldata[:breakPoint], ldata[breakPoint:]


"Find the most optimal correct-pruned tree"
def find_prunned(data_part,f_part):
    monk1train,monkvalue = partition(data_part,f_part)
    dtree = tree.buildTree(monk1train,dataset.attributes)
    prun_list = tree.allPruned(dtree)
    current_correctness = tree.check(dtree,monkvalue)
    for current_tree in prun_list:
        check_correctness = tree.check(current_tree,monkvalue)
        if check_correctness > current_correctness:
            current_correctness = check_correctness
            dtree = current_tree
    return dtree
    

def count_error(training_set, test_set):
  fractions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
  stats_of_pruning = dict()
  for fraction in fractions:
      stats_of_pruning[fraction] = []
      "Make the best calculations for an pruned tree"
      for see in range(100):
         pruned = find_prunned(training_set, fraction)
         if pruned != NULL:
            stats_of_pruning[fraction].append(1-tree.check(pruned,test_set))

  aver = []
  for fraction in stats_of_pruning:
     error_var = np.var(stats_of_pruning[fraction])
     error_mean = np.mean(stats_of_pruning[fraction])
     aver.append(error_mean)
     print(f"Fractions of dataset that are used for the training: {fraction}")
     print(f"The Mean error of {len(stats_of_pruning[fraction])}: {mean_error:.3f}")
     print(f"The Variance of the errors: {error_variance:.3f}")
  return stats_of_pruning,aver 

     
stats1,aver1 = count_error(dataset.monk1,dataset.monk1test)
pt.plot(stats1, aver1)
pt.title("Average of the errors of the prunned tree")
pt.xlabel("The Fractions of dataset for the training")
pt.ylabel("The Average error")  

stats2,aver2 = count_error(dataset.monk3,dataset.monk3test)
pt.plot(stats2, aver2)
pt.title("Average of the errors of the prunned tree")
pt.xlabel("The Fractions of dataset for the training")
pt.ylabel("The Average error")


