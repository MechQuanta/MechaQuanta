import numpy as np
R_1D = np.linspace(-2,2,10)
Z_1D = np.linspace(-1,1,5)
R,Z = np.meshgrid(R_1D,Z_1D,indexing='ij')
print(R,", this is R and ",Z, "this is Z")
R1,Z1 = np.meshgrid(R_1D,Z_1D)
print(R1,", this is R1 and ",Z1, "this is Z1")
