import numpy as np
from numpy import cos, sin, pi
import matplotlib.pyplot as plt

figure = plt.figure()
plt.title("Problem 3B")

S = 10
r = 4

theta = [theta for theta in np.arange(0, S, 0.01)]
X = [
        r * cos(2*pi*theta)/S
        for theta in theta
    ]
Y = [
        r * sin(4*pi*theta)/S
        for theta in theta
    ]

# Plot the results
plt.plot(X, Y)
plt.savefig("./imgs/prob3_b.png")