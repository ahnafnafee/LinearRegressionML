import cv2
from cv2 import VideoWriter, VideoWriter_fourcc
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from collections import Counter
import glob
import numpy as np
import math
import operator
from scipy.linalg import svd
import scipy.stats as ss
import matplotlib.pyplot as plt
import scipy.linalg as la
from matplotlib.image import imread
from enum import Enum


class Sex(Enum):
    male, female = range(2)

class Smoker(Enum):
    no, yes = range(2)
    
class Region(Enum):
    southwest, southeast, northwest, northeast = range(4)


np.random.seed(0)
imp_data = np.genfromtxt('insurance.csv', delimiter=',', encoding='utf8', dtype=np.str)
# dataN = data.astype(np.float)

# Separates header and data
feature_name, data = np.vsplit(imp_data, [1])

# x, y = np.hsplit(data, [-1])

n = len(data)

# Rows shuffled
np.random.shuffle(data)

# Calculates array index for splitting
spltIdx = int((2/3)*n)

# Training-validation data split
data_train, data_test = data[:spltIdx,:], data[spltIdx:,:]

x_tr, y_tr = np.hsplit(data_train, [-1])

def preprocessCat(dataMain):
    '''Changes categorical features to enumerated ones'''

    data = np.copy(dataMain)

    for i in range(len(data)):
        for s in Sex:
            if s.name == data[i][1]:
                data[i][1] = s.value
                break
        
        for sm in Smoker:
            if sm.name == data[i][4]:
                data[i][4] = sm.value
                break

        for r in Region:
            if r.name == data[i][5]:
                data[i][5] = r.value
                break

    return data

def lse(x,y, addBias):
    '''Calculates weights using LSE'''

    dataX = x.astype(np.float64)
    dataY = y.astype(np.float64)

    if addBias:
        biasF = np.ones(((len(dataX)), 1))
        X = np.hstack((biasF, dataX))
    else:
        X = np.array(dataX)


    dotX = (1/len(x)) * np.linalg.inv(np.dot(X.T,X))
    dotY = np.dot(X.T,dataY)

    w = np.dot(dotX, dotY)

    return w



def rmse(w, x, y):
    '''Calculates RMSE'''

    x = x.astype(np.float64)
    y = y.astype(np.float64)

    Y = np.zeros(shape=((len(y)),1))

    for j in range(len(y)):
        Y[j] =  (w[1] * x[j][0]) + (w[2] * x[j][1]) + (w[3] * x[j][2]) + (w[4] * x[j][3]) + (w[5] * x[j][4]) + (w[6] * x[j][5]) + w[0]

    return np.sqrt(np.mean((Y-y)**2))

# Change categorical features to enumerated ones (NO BIAS)
x_tr_preC = preprocessCat(data_train)
w_tr_preC = lse(x_tr_preC, y_tr, False)
rmse_tr_preC = rmse(w_tr_preC, x_tr_preC, y_tr)



# for j in range(len(y_train)):
#     Y[j] = w_t_preC[1]*x[j] + w[0])

print(w_tr_preC)
print(rmse_tr_preC)


