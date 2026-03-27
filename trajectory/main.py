import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.measure import label

def centroid(labeled, label=1):
    ys, xs = np.where(labeled == label)
    return np.mean(ys), np.mean(xs)

def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

lst = []
path = Path("out")

for i in range(100):
    image = np.load(path / f"h_{i}.npy")
    labeled = label(image)

    current_centers = []
    for lb in range(1, np.max(labeled) + 1):
        current_centers.append(centroid(labeled, lb))

    if i == 0:
        for center in current_centers:
            lst.append([center])
    else:
        used_indices = set()

        for track in lst:
            last_point = track[-1]

            closest_index = 0
            min_distance = distance(last_point, current_centers[0])

            for index, center in enumerate(current_centers):
                dist = distance(last_point, center)
                if dist < min_distance:
                    min_distance = dist
                    closest_index = index

            if closest_index not in used_indices:
                track.append(current_centers[closest_index])
                used_indices.add(closest_index)

        for index, center in enumerate(current_centers):
            if index not in used_indices:
                lst.append([center])
                
for track in lst:
    ys = [p[0] for p in track]
    xs = [p[1] for p in track]
    plt.plot(xs, ys, marker="o")

plt.xlabel("X")
plt.ylabel("Y")
plt.show()