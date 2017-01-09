
import numpy as np

def cos(x, y):
    return np.dot(x,y) / (np.sqrt(np.dot(x,x) * np.dot(y,y)))
