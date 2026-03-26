import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.measure import label

def centroid(labeled, label = 1):
    ys, xs, = np.where(labeled==label)
    cy = np.mean(ys)
    cx = np.mean(xs)
    return cy, cx

lst = []
path = Path("out")
for i in range(100):
    image = np.load(path/f"h_{i}.npy")
    labeled = label(image)
    for lb in range(1, np.max(labeled)+1):
        cy, cx = centroid(labeled, lb)
        lst.append((cy, cx))
x = [x for y,x in lst]
y = [y for y,x in lst]
plt.scatter(x,y)
plt.xlabel("X")
plt.ylabel("Y")
plt.show() 
