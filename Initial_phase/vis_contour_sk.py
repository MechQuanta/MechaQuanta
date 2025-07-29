import numpy as np
import matplotlib.pyplot as plt
from skimage import measure

# Create a 2D Gaussian bump (like a Tokamak flux map)
x, y = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
psi = np.exp(-5 * (x**2 + y**2))  # flux shape

# Find contours at 10 levels
levels = np.linspace(0.1, 0.9, 5)
for level in levels:
    contours = measure.find_contours(psi, level)
    for contour in contours:
        plt.plot(contour[:, 1], contour[:, 0], label=f'ψ={level:.2f}')
    for contour in contours:
    	print(contour[:,0],":contour[:,0]")
    	print(contour[:,1],":contour[:,1]")

plt.title('Magnetic Surfaces from ψ')
plt.gca().invert_yaxis()  # match image coordinates
plt.axis('equal')
plt.legend()
plt.show()

