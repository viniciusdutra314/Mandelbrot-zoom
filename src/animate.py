import numpy as np
import matplotlib.pyplot as plt

x=np.load("output.npz")["1"]
plt.imshow(x,cmap="hot")
plt.savefig("output.png")