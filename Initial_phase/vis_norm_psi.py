# Re-import libraries after state reset
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon

# Limiter shape as provided
limiter_shape = np.array([
    [ 1.265,  1.085],
    [ 1.608,  1.429],
    [ 1.683,  1.431],
    [ 1.631,  1.326],
    [ 1.578,  1.32 ],
    [ 1.593,  1.153],
    [ 1.626,  1.09 ],
    [ 2.006,  0.773],
    [ 2.233,  0.444],
    [ 2.235,  0.369],
    [ 2.263,  0.31 ],
    [ 2.298,  0.189],
    [ 2.316,  0.062],
    [ 2.316, -0.062],
    [ 2.298, -0.189],
    [ 2.263, -0.31 ],
    [ 2.235, -0.369],
    [ 2.233, -0.444],
    [ 2.006, -0.773],
    [ 1.626, -1.09 ],
    [ 1.593, -1.153],
    [ 1.578, -1.32 ],
    [ 1.631, -1.326],
    [ 1.683, -1.431],
    [ 1.608, -1.429],
    [ 1.265, -1.085],
    [ 1.265,  1.085]
])

def compute_KSTAR_limiter_mask(RR, ZZ, limiter_shape, min_value=0.05):
    def convert_coord_index(RR, ZZ, points_arr):
        indices_arr = []
        for point in points_arr:
            x, y = point
            idx_x, idx_y = 0, 0
            nx, ny = RR.shape

            for idx in range(nx - 1):
                if RR[0, idx] <= x and RR[0, idx + 1] > x:
                    idx_x = idx
                    break

            for idx in range(ny - 1):
                if ZZ[idx, 0] <= y and ZZ[idx + 1, 0] > y:
                    idx_y = idx
                    break

            indices_arr.append([idx_y, idx_x])  # reverse for (row, col)
        return np.array(indices_arr)

    mask = np.ones_like(RR, dtype=np.float32) * min_value
    contour = convert_coord_index(RR, ZZ, limiter_shape)
    rr, cc = polygon(contour[:, 0], contour[:, 1], mask.shape)
    mask[rr, cc] = 1
    return mask

# Generate R, Z grid
R_vals = np.linspace(1.2, 2.4, 200)
Z_vals = np.linspace(-1.6, 1.6, 300)
RR, ZZ = np.meshgrid(R_vals, Z_vals)

# Compute mask
mask = compute_KSTAR_limiter_mask(RR, ZZ, limiter_shape)

# Plot the mask
plt.figure(figsize=(8, 10))
plt.pcolormesh(RR, ZZ, mask, shading='auto',cmap='coolwarm')
plt.colorbar(label="Limiter Mask Value")
plt.plot(limiter_shape[:,0], limiter_shape[:,1], 'k--', label="Limiter Boundary")
plt.title("Limiter Mask on (R,Z) Grid")
plt.xlabel("R (m)")
plt.ylabel("Z (m)")
plt.legend()
plt.axis("equal")
plt.grid(True)
plt.tight_layout()
plt.savefig("limiter_boundary.png")
plt.show()

