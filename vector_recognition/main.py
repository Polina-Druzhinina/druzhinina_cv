import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.io import imread
from pathlib import Path

save_path = Path(__file__).parent

def count_holes(region):
    shape = region.image.shape
    new_image = np.zeros((shape[0]+2, shape[1]+2))
    new_image[1:-1, 1:-1] = region.image
    new_image = np.logical_not(new_image) #чтобы считать дыры
    labeled = label(new_image)
    return np.max(labeled) -1

def symmetry(region, transpose = False):
    image = region.image
    if transpose:
        image = image.T
    shape = image.shape
    top = image[:shape[0]//2]
    if shape[0] % 2 != 0:
        bottom = image[shape[0]//2+1:]
    else:
        bottom = image[-shape[0]//2:]
    bottom = bottom[::-1] #np.flipud
    result = bottom == top
    return result.sum() / result.size

def extractor(region):
    holes = count_holes(region)
    eccentricity = region.eccentricity
    v_sym = symmetry(region)
    h_sym = symmetry(region, transpose=True)
    return np.array([holes, eccentricity, v_sym, h_sym])

def classificator(region, templates):
    features = extractor(region)
    result = ""
    min_d = 10**6
    for symbol, t in templates.items():
        d = ((t-features)**2).sum() ** 0.5
        if d < min_d:
            result = symbol
            min_d = d
    return result


template = imread("alphabet/alphabet-small.png")[:,:,:-1]
template = template.sum(2) 
binary = template != 765 
labeled = label(binary)
props = regionprops(labeled)

templates ={}
for region, symbol in zip(props, ["8", "O", "A", "B", "1", "W", "X", "*", "/", "-"]):
    templates[symbol] = extractor(region)

image = imread("alphabet/alphabet.png")[:,:,:-1]
abinary = image.mean(2) > 0
alabeled = label(abinary)
aprops = regionprops(alabeled)
result = {}
image_path = save_path / "out_vector"
image_path.mkdir(exist_ok=True)
plt.figure(figsize=(5,7))
for region in aprops:
    symbol = classificator(region, templates)
    if symbol not in result:
        result[symbol] = 0
    result[symbol] += 1
    plt.cla()
    plt.title(f"Class - '{symbol}'")
    plt.imshow(region.image)
    plt.savefig(image_path/f"image_{region.label}.png")
print(result)

plt.imshow(alabeled)
plt.show()