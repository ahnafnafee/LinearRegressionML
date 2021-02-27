import numpy as np
import math
import operator
from scipy.linalg import svd
import scipy.stats as ss
import matplotlib.pyplot as plt
import scipy.linalg as la
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from collections import Counter
import glob
from matplotlib.image import imread


data = np.genfromtxt('insurance.csv', delimiter=',')
# Separates the class label from the observable data to form matrices Y and X
Y, X = np.vsplit(data, [1])


