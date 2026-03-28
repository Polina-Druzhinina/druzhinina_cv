import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage.morphology import (opening)

image = np.load("stars.npy")
labeled = label(image)

struct = np.ones((3,3))
count = 0
for lb in range(1, labeled.max()+1):
    image_lb = labeled==lb
    count_before = np.sum(image_lb)
    ob = opening(image_lb, struct)
    count_after = np.sum(ob)
    if(count_after < count_before):
        count += 1
print(f"Количество звезд: {count}")
plt.imshow(labeled)
plt.show()
