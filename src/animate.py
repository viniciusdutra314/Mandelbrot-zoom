import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

npz = np.load("output.npz")
arrays = [npz[f] for f in npz.files]  

cmap = plt.get_cmap("twilight")

fig, ax = plt.subplots()
im = ax.imshow(arrays[0], cmap=cmap.reversed())
ax.set_title("Frame 0")

def update(frame):
    im.set_array(arrays[frame])
    ax.set_title(f"Frame {frame}")
    return [im]

ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(arrays),
    interval=1/30, 
    blit=True
)

ani.save("animation.mp4", writer="ffmpeg",fps=15)
plt.show()
