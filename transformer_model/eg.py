# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Visualizing the convergence of force residuals using VMEC output quantities."""
from pathlib import Path
import matplotlib.pyplot as plt
import vmecpp
import os

# Use the correct data path
data_path = Path.home() / "vmecpp" / "examples" / "data"
input_file = data_path / "input.w7x"

# Check if file exists
if not input_file.exists():
    raise FileNotFoundError(f"Input file not found at {input_file}")

input = vmecpp.VmecInput.from_file(input_file)

# Run the VMEC solver to compute equilibrium
output = vmecpp.run(input)

def plot_force_residuals(ax):
    # Plot force residuals
    ax.plot(output.wout.force_residual_r, label="Force residual ($R$)")
    ax.plot(output.wout.force_residual_z, label="Force residual ($Z$)")
    ax.plot(output.wout.force_residual_lambda, label=r"Force residual ($\lambda$)")
    ax.plot(output.wout.fsqt, "k", label="Force residual (total)")
    
    # Plot additional quantities if available
    if hasattr(output.wout, 'lfreeb') and output.wout.lfreeb:
        ax.plot(output.wout.delbsq, label=r"$\Delta B^2$")
    
    # Plot tolerance line
    ax.axhline(y=output.wout.ftolv, color="red", linestyle="dashed", label="Tolerance")
    
    # Plot restart reasons if available
    if hasattr(output.wout, 'restart_reasons'):
        restart_labels_added = set()  # To avoid duplicate labels
        for i, reason in output.wout.restart_reasons:
            label = "Restart: " + reason.name if reason.name not in restart_labels_added else None
            if label:
                restart_labels_added.add(reason.name)
            ax.axvline(
                i,
                ymin=0,  # Start from bottom of plot
                ymax=1,  # Go to top of plot (normalized coordinates)
                color="gray",
                alpha=0.7,
                label=label,
            )
    
    ax.set_yscale("log")

# Create the main plot
fig, ax = plt.subplots(figsize=(10, 6))
plot_force_residuals(ax)
ax.set_xlabel("Iteration")
ax.set_ylabel("Force residual")
ax.legend()
ax.set_title("Force Residual Convergence")

# Create an inset plot in the bottom left corner
axins = ax.inset_axes([0.25, 0.5, 0.25, 0.4])
plot_force_residuals(axins)

# Set limits for inset plot with bounds checking
max_iter = min(30, len(output.wout.fsqt))
if max_iter > 0:
    axins.set_xlim(0, max_iter)
    
    # Calculate y-limits safely
    y_min_vals = []
    if len(output.wout.force_residual_r) > 0:
        y_min_vals.append(output.wout.force_residual_r[:max_iter].min())
    if len(output.wout.force_residual_z) > 0:
        y_min_vals.append(output.wout.force_residual_z[:max_iter].min())
    if len(output.wout.force_residual_lambda) > 0:
        y_min_vals.append(output.wout.force_residual_lambda[:max_iter].min())
    
    if y_min_vals:
        y_min = min(y_min_vals)
        y_max = output.wout.fsqt[:max_iter].max() * 1.1
        axins.set_ylim(y_min, y_max)

# Remove x-axis labels from inset
axins.set_xticklabels([])

# Add zoom indication
ax.indicate_inset_zoom(axins)

plt.tight_layout()
plt.show()
