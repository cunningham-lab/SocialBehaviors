import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

def get_colors_and_cmap(K):
    if K <= 10:
        colors = np.array(plt.cm.tab10.colors)[:K]
        colors_list = colors.tolist()
    elif K <= 20:
        colors = np.array(plt.cm.tab20.colors)[:K]
        colors_list = colors.tolist()
    else:
        vals = np.linspace(0, 1, 10 * K)[10 * np.arange(K)]
        colors = plt.cm.tab20(vals)
        colors_list = colors.tolist()

    cvals = np.arange(K)

    norm=plt.Normalize(min(cvals),max(cvals))
    tuples = list(zip(map(norm,cvals), colors_list))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples, K)

    return colors, cmap


colors, cm = get_colors_and_cmap(10)
colors_to_plot = [colors[1], colors[2], colors[3], colors[4], colors[5], colors[5], colors[5], colors[0], colors[1], colors[2]]

Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
Y = X
XY = np.column_stack((X, Y))
U = np.ones((10,))
V = np.ones((10,))
dXY = np.column_stack((U, V))

fig, ax = plt.subplots(1, 1)
ax.set_xlim(-1, 20)
ax.set_ylim(-5, 20)
ax.axis('equal')
Q = ax.quiver([], [], [], [], scale=1, units="xy", cmap=cm, color=colors_to_plot)


# colors=[10, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]

def update_quiver(num, Q, XY, dXY, colors):
    """updates the horizontal and vertical vector components by a
    fixed increment on each frame
    """

    U = dXY[:num + 1, 0]
    V = dXY[:num + 1, 1]

    Q.set_offsets(XY[:num + 1])
    Q.set_UVC(U, V)
    return Q,


# you need to set blit=False, or the first set of arrows never gets
# cleared on subsequent frames
anim = animation.FuncAnimation(fig, update_quiver, frames=len(XY), fargs=(Q, XY, dXY, colors),
                               interval=10, blit=False)
anim.save("quiver.mp4", writer=writer)