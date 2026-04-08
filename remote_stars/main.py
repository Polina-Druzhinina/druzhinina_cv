import numpy as np
import matplotlib.pyplot as plt
import socket
from skimage.measure import label,regionprops

host = "84.237.21.36"
port = 5152

def recvall(sock, nbytes):
    data = bytearray()
    while len(data) < nbytes:
        packet = sock.recv(nbytes - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

plt.ion()
plt.figure()
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.connect((host, port))
    sock.send(b"124ras1")
    print(sock.recv(10))
    beat = b"hope"
    while beat != b"yep":
        sock.send(b"get")
        bts = recvall(sock, 40002)

        im1 = np.frombuffer(bts[2:], dtype="uint8")
        im1 = im1.reshape(200,200)
        
        labeled = label(im1 > 0)
        coor = []
        for region in regionprops(labeled):
            coor.append(region.centroid)
        (y1,x1) = coor[0]
        (y2,x2) = coor[1]
        dist = round(((x2-x1)**2 + (y2-y1)**2)**0.5, 1)
        sock.send(f"{dist}".encode())
        print(sock.recv(10))
        
        plt.clf()
        plt.imshow(im1)
        plt.pause(2)
        sock.send(b"beat")
        beat = sock.recv(10)