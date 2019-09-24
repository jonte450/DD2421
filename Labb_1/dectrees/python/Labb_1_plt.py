#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 13:59:34 2019

@author: jonteyh
"""
from matplotlib import pyplot as plt

"Monk-1"
list_means = [0.23,0.21,0.189,0.168,0.152,0.150]
fractions_vector = [0.3,0.4,0.5,0.6,0.7,0.8]
list_variance_1  = [0.00198,0.00198,0.00198,0.00198,0.00198,0.00198]

"Monk-3"
list_means3 = [0.092, 0.074,0.059,0.052,0.051,0.046]
list_variance_3  = [0.002,0.0015,0.001,0.001,0.001,0.001]
plt.title("Average error of prunned tree data_1")
plt.xlabel("Fractions")

plt.plot(fractions_vector, list_means)
plt.errorbar(fractions_vector,list_means, yerr=list_variance_1)
plt.legend(["error"])
plt.show()

plt.title("Average error of prunned tree data_3")
plt.xlabel("Fractions")

plt.plot(fractions_vector, list_means3)
plt.errorbar(fractions_vector,list_means3, yerr=list_variance_3)
plt.legend(["error"])
plt.show()