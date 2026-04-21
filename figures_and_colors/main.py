import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.io import imread
from skimage.color import rgb2hsv, rgb2gray

image = imread("./data/balls_and_rects.png")
#кол-во фигур
hsv = rgb2hsv(image)
v = hsv[:,:,2]
labeled = label(v>0.3)
print(f"Количество всех фигур на изображении: {labeled.max()}")

#по оттенкам 
hsv = rgb2hsv(image)
h = hsv[:,:,0]

figure = {"circle":{}, "rectangle":{}}

for region in regionprops(labeled):
    mask = labeled == region.label
    mean_hue = h[mask].mean()
    circularity = (4 * 3.14 * region.area)/(region.perimeter**2)
    extent = region.extent
    #print(circularity, extent)
    if extent < 0.9 and circularity > 0.85:
        if mean_hue not in figure["circle"]:
            figure["circle"][mean_hue] = 0
        figure["circle"][mean_hue] += 1
    else:
        if mean_hue not in figure["rectangle"]:
            figure["rectangle"][mean_hue] = 0
        figure["rectangle"][mean_hue] += 1

for shape in figure:
    print(f"{shape}:")
    for color in figure[shape]:
        print(f"{color}: {figure[shape][color]}")
    print()

plt.imshow(h, cmap="gray")
plt.show()
