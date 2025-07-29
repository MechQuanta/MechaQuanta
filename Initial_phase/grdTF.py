import numpy as np
from scipy import interpolate

# Create a grid
R1D = np.linspace(1, 2, 5)  # [1. , 1.25, 1.5 , 1.75, 2. ]
Z1D = np.linspace(-1, 1, 3) # [-1. , 0. , 1. ]

RR, ZZ = np.meshgrid(R1D, Z1D)  # 3 rows, 5 columns
F = RR**2 + ZZ**2  # function values at each (R, Z)

print(RR)
print(ZZ)
print(F)

