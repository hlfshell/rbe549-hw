import numpy as np
from numpy import cos, sin, pi
import matplotlib.pyplot as plt

figure = plt.figure()
plt.title("Problem 2C")

S = 10
r = 4
x0 = 1
y0 = 1

theta = [theta for theta in np.arange(0, S, 0.01)]
X = [
        x0 + r * cos(2*pi*theta)/S
        for theta in theta
    ]
Y = [
        y0 + r * sin(2*pi*theta)/S
        for theta in theta
    ]

# Plot the results
plt.plot(X, Y)
plt.savefig("./imgs/prob2_c.png")