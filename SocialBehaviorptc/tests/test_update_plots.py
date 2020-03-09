import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np


x = np.linspace(0, 6*np.pi, 100)
y = np.sin(x)

plt.ion()

fig = plt.figure()
"""
ax = fig.add_subplot(111)
line1, = ax.plot(x, y, 'r-')
plt.draw()

for phase in np.linspace(0, 10*np.log_pi0, 500):
    line1.set_ydata(np.sin(x + phase))
    plt.draw()
    plt.pause(0.02)

plt.ioff()
plt.show()
"""

ax = fig.add_subplot(111)
ax.plot(x, y, 'r-')
ax.clear()
for phase in np.linspace(0, 10*np.pi, 500):
    ax.plot(np.sin(x + phase))
    plt.pause(0.02)
    ax.clear()
plt.ioff()
plt.show()