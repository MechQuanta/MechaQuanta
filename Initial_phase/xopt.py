import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from numpy.linalg import inv

# -----------------------------------
# 1. Create Tokamak-like psi profile
# -----------------------------------
r1d = np.linspace(1.0, 2.0, 200)
z1d = np.linspace(-1.0, 1.0, 200)
R, Z = np.meshgrid(r1d, z1d)

R0 = 1.5
a = 0.5
kappa = 1.7

psi = ((R - R0) / a)**2 + (Z / (kappa * a))**2
psi += 0.05 * np.exp(-((R - (R0 + 0.4*a))**2 + (Z + 0.6*a)**2) / (0.1*a)**2)

# Interpolation
f = interpolate.RectBivariateSpline(r1d, z1d, psi)

# -----------------------------------
# 2. Compute Bp^2
# -----------------------------------
Bp2 = (f(r1d, z1d, dx=1, grid=True)**2 + f(r1d, z1d, dy=1, grid=True)**2) / R**2

dR = r1d[1] - r1d[0]
dZ = z1d[1] - z1d[0]
radius_sq = 9 * (dR**2 + dZ**2)

J = np.zeros([2, 2])
xpoint = []
opoint = []

nx, ny = Bp2.shape

# -----------------------------------
# 3. Find O-points and X-points
# -----------------------------------
for i in range(2, nx - 2):
    for j in range(2, ny - 2):
        if (
            (Bp2[i, j] < Bp2[i+1, j+1]) and (Bp2[i, j] < Bp2[i+1, j]) and (Bp2[i, j] < Bp2[i+1, j-1]) and
            (Bp2[i, j] < Bp2[i-1, j+1]) and (Bp2[i, j] < Bp2[i-1, j]) and (Bp2[i, j] < Bp2[i-1, j-1]) and
            (Bp2[i, j] < Bp2[i, j+1]) and (Bp2[i, j] < Bp2[i, j-1])
        ):
            R0c = R[i, j]
            Z0c = Z[i, j]
            R1 = R0c
            Z1 = Z0c

            count = 0
            while True:
                # Use .ev() for single points
                Br = -f.ev(R1, Z1, dy=1) / R1
                Bz =  f.ev(R1, Z1, dx=1) / R1

                if Br**2 + Bz**2 < 1e-6:
                    d2dr2 = (psi[i+2,j] - 2*psi[i,j] + psi[i-2,j]) / (2*dR)**2
                    d2dz2 = (psi[i,j+2] - 2*psi[i,j] + psi[i,j-2]) / (2*dZ)**2
                    d2drdz = (
                        (psi[i+2,j+2] - psi[i+2,j-2]) / (4*dZ) -
                        (psi[i-2,j+2] - psi[i-2,j-2]) / (4*dZ)
                    ) / (4*dR)
                    
                    D = d2dr2 * d2dz2 - d2drdz**2

                    if D < 0:
                        xpoint.append((R1, Z1, f.ev(R1, Z1)))
                    else:
                        opoint.append((R1, Z1, f.ev(R1, Z1)))
                    break

                J[0,0] = -Br/R1 - f.ev(R1, Z1, dy=1, dx=1)/R1
                J[0,1] = -f.ev(R1, Z1, dy=2)/R1
                J[1,0] = -Bz/R1 + f.ev(R1, Z1, dx=2)/R1
                J[1,1] = f.ev(R1, Z1, dx=1, dy=1)/R1

                d = inv(J).dot([Br, Bz])
                R1 -= d[0]
                Z1 -= d[1]

                count += 1
                if ((R1 - R0c)**2 + (Z1 - Z0c)**2 > radius_sq) or (count > 100):
                    break

# -----------------------------------
# 4. Plotting
# -----------------------------------
fig, ax = plt.subplots(figsize=(9, 7))

levels = np.linspace(np.min(psi), np.max(psi), 50)
contour = ax.contour(R, Z, psi, levels=levels, cmap='plasma')

cbar = fig.colorbar(contour)
cbar.set_label('Poloidal Flux $\psi$')

if opoint:
    opoint = np.array(opoint)
    ax.plot(opoint[:, 0], opoint[:, 1], 'ro', label='O-point', markersize=10)
    for idx, (r, z, psi_val) in enumerate(opoint):
        ax.text(r, z, f'O{idx}', color='red', fontsize=8, ha='left', va='bottom')

if xpoint:
    xpoint = np.array(xpoint)
    ax.plot(xpoint[:, 0], xpoint[:, 1], 'bx', label='X-point', markersize=10)
    for idx, (r, z, psi_val) in enumerate(xpoint):
        ax.text(r, z, f'X{idx}', color='blue', fontsize=8, ha='left', va='bottom')

ax.set_xlabel('R [m]')
ax.set_ylabel('Z [m]')
ax.set_title('Tokamak-like Poloidal Flux Contours with O-points and X-points')
ax.legend()
ax.axis('equal')
ax.grid(True)
plt.savefig("xopts.png")
plt.show()

