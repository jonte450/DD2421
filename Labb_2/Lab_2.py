# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import random
import math
from scipy.optimize import minimize
import matplotlib.pyplot as plt



"""Generating the datasets"""
np.random.seed(100)
classA = np.concatenate((np.random.rand(10,2) * 0.2 +[1.5, 0.5],np.random.randn(10, 2) * 0.2 + [-1.5, 0.5]))
classB = np.random.randn(20, 2) * 0.2 + [0.0 , -0.5]
inputs = np.concatenate((classA, classB))
targets = np.concatenate((np.ones(classA.shape[0]),-np.ones(classB.shape[0])))
N = inputs.shape[0] #Number of rows Samples
permute = list(range(N))
random.shuffle(permute)
inputs = inputs[permute, :]
targets = targets[permute]




"""The kernel function"""

def linear_kernel(x,y):
    return np.dot(x,y)

def polynomial_kernel(x1, x2):
  p = 3
  # p = 3
  return np.power(np.dot(x1, x2) + 1, p)

def radial_kernel(x, y, sigma=0.1):
    diff = np.subtract(x, y)
    return math.exp((-np.dot(diff, diff)) / (2 * sigma* sigma))

"""Choose kernel"""
kernel = linear_kernel
  
"""Help-function for calculate the sum of the target function and Kernel"""
def cal_part_two():
    step2 = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
           step2[i,j] = targets[i]*targets[j]*kernel([inputs[i,0],inputs[i,1]],[inputs[j,0],inputs[j,1]])  
    return step2

"""Obejctive funtction"""
def objective(alfa):
    sum_1 = 0
    step_list_2 = cal_part_two()
    for i,alpha_i in enumerate(alfa):
     for j,alpha_j in enumerate(alfa): 
         sum_1 += 0.5 * alpha_i * alpha_j * step_list_2[i,j]
    sum_2 = np.sum(alfa)
    resultat = sum_1 - sum_2
    return resultat
 
"""Zero-function"""
def zero_function(alpha):
    return np.dot(alpha,targets)

"""Minimize-function"""
def call_minimizefuntion():
    mini = minimize(objective, start, bounds=B, constraints={'type': 'eq', 'fun': zero_function})
    return mini

"""Non-Zero values"""
def extract_zeros(alpha):
    extracted = []
    for value,alpha_check in enumerate(alpha):
         if alpha_check > 1.e-5:
             extracted.append((alpha[value],inputs[value][0],inputs[value][1],targets[value]))
             plt.plot(inputs[value][0], inputs[value][1], 'g+')

    return extracted      

"""Check for Slack and calculate b"""
def calculate_b(alpha,value):
    b_sum = 0
    for calc in range(N):
        b_sum += alpha[calc]*targets[calc]*kernel(inputs[calc],inputs[value])
    b_sum -= targets[value]
    return b_sum         
    
def check_value(alpha):
    b = 0
    for check in range(N):
     if check > 1.e-5 and check < C:
         b = calculate_b(alpha,check)
    return b


"""Indicate function"""
def indicator(svm,x_points,y_points,kernel):
    ind_sum = 0
    for add in range(len(svm)):
        ind_sum += svm[add][0]*svm[add][3]*kernel([x_points,y_points],[svm[add][1],svm[add][2]])
    ind_sum-=b
    return ind_sum


"""Begin Processing the data"""
C = 100000
B =[(0, C) for b in range(N)]

start = np.zeros(N);

return_data =call_minimizefuntion();
alphas = return_data['x'];


"""Take away zeros"""
svm = []
svm = extract_zeros(alphas);


"""Calculate the b"""
b = check_value(alphas);



plt.plot([p[0] for p in classA],
         [p[1] for p in classA],
         'b.')

plt.plot([p[0] for p in classB],
         [p[1] for p in classB],
         'r.')

plt.axis('equal')
plt.savefig('svmplot.pdf')  


xgrid = np.linspace(-5, 5)
ygrid = np.linspace(-4 , 4)

grid = np.array([[indicator(svm,x,y,kernel)
                     for x in xgrid]
                     for y in ygrid])

plt.contour(xgrid, ygrid, grid, 
           (-1.0, 0.0 , 1.0),
            colors=('red','black','green'),
            linewidths=(1, 3, 1))  

plt.show()
    
