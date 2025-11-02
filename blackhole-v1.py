# Revised visualization: true spherical lens (not ellipse), no blue, adds orange->yellow accretion disk,
# and warps the GRID beneath the lens (so you actually see gravitational lensing of the grid).
# Vectorized and robust to matplotlib versions. Run in a graphical environment.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Parameters
grid_z_level = -5.0
grid_size = 20
grid_spacing = 1.0

sphere_radius = 3.0
sphere_z_position = grid_z_level + sphere_radius + 2.0  # center of black hole above grid

lens_radius = sphere_radius * 3.0   # influence radius (larger gives stronger/wider warp)
distortion_factor = 0.6             # stronger distortion as user requested (0.6)

# Resolutions
grid_res = int(2 * grid_size / grid_spacing + 1)
lens_u_res = 120
lens_v_res = 120
sphere_u_res = 100
sphere_v_res = 100
disk_radial_res = 200
disk_angular_res = 200

# Create grid (X, Y) at constant Z = grid_z_level
x = np.arange(-grid_size, grid_size + grid_spacing, grid_spacing)
y = np.arange(-grid_size, grid_size + grid_spacing, grid_spacing)
X, Y = np.meshgrid(x, y)
Z = np.full_like(X, grid_z_level)

# Warp the grid to simulate lensing
# Compute radial distance in XY plane from black hole center (0,0)
r = np.sqrt(X**2 + Y**2)
# Influence mask
mask = r < lens_radius
# Avoid divide-by-zero
r_safe = r.copy()
r_safe[r_safe == 0] = 1.0

# Direction unit vectors from center
ux = X / r_safe
uy = Y / r_safe

# Base distortion magnitude decreases with radius; stronger near center
raw = (1.0 - (r / lens_radius))
raw = np.clip(raw, 0.0, 1.0)
# Add an angular twist pattern to create believable bending
angle = np.arctan2(Y, X)
twist = np.sin(angle * 3.0) * 0.3

# Displacement magnitude (vectorized)
disp = distortion_factor * raw * (1.0 + 0.6 * np.sin((r / (lens_radius+1e-9)) * np.pi * 2.0))  # radial variation
# Apply only where mask true
DX = ux * disp * (lens_radius * 0.35)
DY = uy * disp * (lens_radius * 0.35)
# Add a small tangential swirl component for realism
DX += -uy * twist * disp * (lens_radius * 0.08)
DY += ux * twist * disp * (lens_radius * 0.08)

# Produce warped grid coordinates (only warp where inside influence)
X_warp = X.copy()
Y_warp = Y.copy()
X_warp[mask] = X[mask] + DX[mask]
Y_warp[mask] = Y[mask] + DY[mask]

# Lensing surface (a translucent shell around the black hole) - color neutral (no blue)
u_lens = np.linspace(0, 2 * np.pi, lens_u_res)
v_lens = np.linspace(0, np.pi, lens_v_res)
U_l, V_l = np.meshgrid(u_lens, v_lens, indexing='xy')  # (v_res, u_res)

x_lens = (lens_radius * np.cos(U_l) * np.sin(V_l))
y_lens = (lens_radius * np.sin(U_l) * np.sin(V_l))
z_lens = (lens_radius * np.cos(V_l)) + sphere_z_position

# Apply a radial-dependent distortion to the lens surface (so it looks warped, not an ellipse)
# Compute distance from sphere center for each lens point
dist_lens = np.sqrt(x_lens**2 + y_lens**2 + (z_lens - sphere_z_position)**2)
raw_l = (1.0 - dist_lens / (lens_radius + 1e-9))
raw_l = np.clip(raw_l, 0.0, 1.0)
# stronger warp near the front (smaller V) to mimic light bending
warp_strength = distortion_factor * (0.6 + 0.8 * np.sin(V_l))
angle_l = np.arctan2(y_lens, x_lens)

x_lens = x_lens + (raw_l * warp_strength) * np.cos(angle_l + 0.5 * np.sin(3 * V_l)) * dist_lens * 0.12
y_lens = y_lens + (raw_l * warp_strength) * np.sin(angle_l + 0.5 * np.sin(3 * V_l)) * dist_lens * 0.12
z_lens = z_lens + (raw_l * warp_strength) * np.cos(V_l) * dist_lens * 0.06

# Black sphere (event horizon)
u_sphere = np.linspace(0, 2 * np.pi, sphere_u_res)
v_sphere = np.linspace(0, np.pi, sphere_v_res)
U_s, V_s = np.meshgrid(u_sphere, v_sphere, indexing='xy')
x_sphere = sphere_radius * np.cos(U_s) * np.sin(V_s)
y_sphere = sphere_radius * np.sin(U_s) * np.sin(V_s)
z_sphere = sphere_radius * np.cos(V_s) + sphere_z_position

# Accretion disk (orange -> yellow)
disk_r_inner = sphere_radius * 1.05
disk_r_outer = sphere_radius * 5.0
rad = np.linspace(disk_r_inner, disk_r_outer, disk_radial_res)
theta = np.linspace(0, 2 * np.pi, disk_angular_res)
R, T = np.meshgrid(rad, theta, indexing='xy')
# Disk coordinates - slight tilt for realism
tilt = np.deg2rad(12)  # 12 degree tilt
X_disk = (R * np.cos(T))
Y_disk = (R * np.sin(T)) * np.cos(tilt) - (R * np.sin(T) * np.sin(tilt) * 0)  # simple tilt on Y
Z_disk = np.full_like(X_disk, sphere_z_position + 0.02)  # slightly above equator

# Colors: gradient from orange to yellow based on radius
from matplotlib import cm
colors = np.empty(X_disk.shape + (4,))
# radial normalized
rn = (R - disk_r_inner) / (disk_r_outer - disk_r_inner)
# make palette: orange -> yellow
for i in range(X_disk.shape[0]):
    colors[i, :, :] = cm.get_cmap('YlOrRd')(1 - rn[i, :])  # reversed to have orange inner, yellow outer

# Plot
fig = plt.figure(figsize=(11, 9))
fig.patch.set_facecolor('black')
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('black')

# Warped grid surface (so you can see lensing)
ax.plot_surface(X_warp, Y_warp, Z, rstride=1, cstride=1, alpha=0.9, color='white', edgecolor='none', linewidth=0)

# Wireframe of warped grid for crisp lines
ax.plot_wireframe(X_warp, Y_warp, Z, rstride=2, cstride=2, color='white', linewidth=0.6, alpha=0.95)

# Lensing translucent shell (neutral color)
ax.plot_surface(x_lens, y_lens, z_lens, rstride=1, cstride=1, alpha=0.18,
                color=(0.9, 0.9, 0.9), edgecolor='none', linewidth=0, antialiased=True)

# Accretion disk (use facecolors)
ax.plot_surface(X_disk, Y_disk, Z_disk, rstride=1, cstride=1, facecolors=colors, shade=False, linewidth=0)

# Black sphere
ax.plot_surface(x_sphere, y_sphere, z_sphere, rstride=1, cstride=1,
                color='black', edgecolor='none', linewidth=0, antialiased=True)

# Labels and style
ax.set_xlabel('X', color='white')
ax.set_ylabel('Y', color='white')
ax.set_zlabel('Z', color='white')
ax.set_title('Black Hole — Accretion Disk + Lensing (no blue)', color='white', pad=16)
ax.tick_params(colors='white', which='both')

# Attempt equal aspect ratio
try:
    ax.set_box_aspect((1, 1, 1))
except Exception:
    # fallback: set symmetric limits based on maximum extent so sphere doesn't look squashed
    max_extent = max(grid_size, lens_radius, disk_r_outer)
    ax.set_xlim(-max_extent, max_extent)
    ax.set_ylim(-max_extent, max_extent)
    ax.set_zlim(grid_z_level - 8, sphere_z_position + lens_radius + 6)

# Clean up default panes if available
try:
    ax.xaxis.pane.set_facecolor((0,0,0,0))
    ax.yaxis.pane.set_facecolor((0,0,0,0))
    ax.zaxis.pane.set_facecolor((0,0,0,0))
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')
except Exception:
    pass

ax.grid(False)
plt.tight_layout()
plt.show()
