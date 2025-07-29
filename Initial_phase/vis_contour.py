import numpy as np
import matplotlib.pyplot as plt

# Grid setup
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)

# Different spreads in x and y directions
sigma_x = 0.5
sigma_y = 1.5

# Asymmetric Gaussian
psi = np.exp(-((X**2)/(sigma_x**2) + (Y**2)/(sigma_y**2)))

# Plot
plt.contourf(X, Y, psi, levels=50, cmap='inferno')
plt.title("Asymmetric Gaussian bump (Elliptical)")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar(label='ψ(x, y)')
plt.gca().set_aspect('equal')
plt.show()

