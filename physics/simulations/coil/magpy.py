import numpy as np
import magpylib as mp

### diameter is 8.25mm
### width = 7.5mm

coil_ssRF = mp.Collection()
coil_NMR = mp.Collection()
turns = 40
current_ssRF = .5
current_NMR = .001
diameter_ssRF = 0.020955 # 0.825 inch
diameter_NMR = 0.018415 # 0.725 inch
width = 0.018415 # .725 inch
for z in np.linspace(-width/2, width/2, turns):
    winding = mp.current.Circle(
        current=current_ssRF,
        diameter=diameter_ssRF,
        position=(0,0,z),
    )
    coil_ssRF.add(winding)

for z in np.linspace(-0.0001, 0.0001, 1):
    winding = mp.current.Circle(
        current=current_NMR,
        diameter=diameter_NMR,
        position=(0,0,0),
    )
    coil_NMR.add(winding)
    

nmr_coil = mp.Collection(coil_ssRF, coil_NMR)
# Coil cross-sections for y-z plot (x=0): each Circle appears as line from (-r,z) to (r,z)
radius_ssRF = diameter_ssRF / 2
radius_NMR = diameter_NMR / 2
steps = 200
import matplotlib.pyplot as plt
# Match figsize to data aspect (y/z extent) to avoid whitespace
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

# Compute field and plot Bx-By field lines in the xy-plane (z=0)
# Use grid matching displayed region (-0.01, 0.01) with fine resolution for dense field lines
x = np.linspace(-0.015, 0.015, steps)
y = np.linspace(-0.015, 0.015, steps)
X, Y = np.meshgrid(x, y)
grid = np.stack([X, Y, np.zeros_like(X)], axis=-1)

B = mp.getB(nmr_coil, grid)
Bx, By = B[:, :, 0], B[:, :, 1]

Bamp = np.linalg.norm(B, axis=2)
Bamp /= np.amax(Bamp)

sp = ax.streamplot(X, Y, Bx, By, density=4, color=Bamp,
    linewidth=np.sqrt(Bamp)*3, cmap='coolwarm',
)

# Plot coil cross-sections (circles in xy-plane at z=0)
theta = np.linspace(0, 2*np.pi, 200)
ax.plot(radius_ssRF*np.cos(theta), radius_ssRF*np.sin(theta), 'k-', lw=2.5, zorder=10, label='ssRF coil')
ax.plot(radius_NMR*np.cos(theta), radius_NMR*np.sin(theta), 'lime', lw=2, ls='--', zorder=11, label='NMR coil')
ax.legend(loc='upper right')

# Figure styling
ax.set(
    title='Bx-By field lines of Coil (xy-plane, z=0)',
    xlabel='x-position (m)',
    ylabel='y-position (m)',
    aspect='equal',
    xlim=(-0.015, 0.015),
    ylim=(-0.015, 0.015),
)
ax.margins(0)
plt.colorbar(sp.lines, ax=ax, label='|B|/|B_max|')

plt.tight_layout(pad=0.5)
# plt.show()
plt.savefig('coil_field_lines_xy.png', dpi=300, bbox_inches='tight', facecolor='white')

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1, figsize=(6, 6))

# Compute field of the coil pair on xy-grid (bx-by plane, z=0)
x = np.linspace(-0.015, 0.015, steps)
y = np.linspace(-0.015, 0.015, steps)
X, Y = np.meshgrid(x, y)
grid = np.stack([X, Y, np.zeros_like(X)], axis=-1)

B = mp.getB(nmr_coil, grid)

# Field at center
B0 = mp.getB(nmr_coil, (0,0,0))
B0amp = np.linalg.norm(B0)

# Homogeneity error
err = np.linalg.norm((B-B0)/B0amp, axis=2)

# Plot error on grid (xy-plane)
sp = ax.contourf(X, Y, err*100)

# Plot coil cross-sections (circles in xy-plane at z=0)
theta = np.linspace(0, 2*np.pi, 200)
ax.plot(radius_ssRF*np.cos(theta), radius_ssRF*np.sin(theta), 'k-', lw=2, zorder=10, label='ssRF coil')
ax.plot(radius_NMR*np.cos(theta), radius_NMR*np.sin(theta), 'lime', lw=2, ls='--', zorder=11, label='NMR coil')
ax.legend(loc='upper right')

# Figure styling
ax.set(
    title='Coil homogeneity error (bx-by plane, z=0)',
    xlabel='x-position (m)',
    ylabel='y-position (m)',
    aspect='equal',
    xlim=(-0.015, 0.015),
    ylim=(-0.015, 0.015),
)
ax.margins(0)
plt.colorbar(sp, ax=ax, label='(% of B0)')

plt.tight_layout(pad=0.5)
# plt.show()
plt.savefig('coil_homogeneity_error_xy.png', dpi=300, bbox_inches='tight', facecolor='white')

# --- Bz-By plane (yz-plane, x=0): field lines ---
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
y = np.linspace(-0.015, 0.015, steps)
z = np.linspace(-0.015, 0.015, steps)
Y, Z = np.meshgrid(y, z)
grid = np.stack([np.zeros_like(Y), Y, Z], axis=-1)

B = mp.getB(nmr_coil, grid)
By, Bz = B[:, :, 1], B[:, :, 2]

Bamp = np.linalg.norm(B, axis=2)
Bamp /= np.amax(Bamp)

sp = ax.streamplot(Y, Z, By, Bz, density=4, color=Bamp,
    linewidth=np.sqrt(Bamp)*3, cmap='coolwarm',
)

# Plot coil cross-sections in yz-plane (x=0): coil1 = rectangle, coil2 = horizontal line at z=0
ax.plot([-radius_ssRF, radius_ssRF], [-width/2, -width/2], 'k-', lw=2.5, zorder=10)
ax.plot([-radius_ssRF, radius_ssRF], [width/2, width/2], 'k-', lw=2.5, zorder=10)
ax.plot([-radius_ssRF, -radius_ssRF], [-width/2, width/2], 'k-', lw=2.5, zorder=10, label='ssRF coil')
ax.plot([radius_ssRF, radius_ssRF], [-width/2, width/2], 'k-', lw=2.5, zorder=10)
ax.plot([-radius_NMR, radius_NMR], [0, 0], 'lime', lw=2, ls='--', zorder=11, label='NMR coil')
ax.legend(loc='upper right')

ax.set(
    title='Bz-By field lines of Coil (yz-plane, x=0)',
    xlabel='y-position (m)',
    ylabel='z-position (m)',
    aspect='equal',
    xlim=(-0.015, 0.015),
    ylim=(-0.015, 0.015),
)
ax.margins(0)
plt.colorbar(sp.lines, ax=ax, label='|B|/|B_max|')
plt.tight_layout(pad=0.5)
# plt.show()
plt.savefig('coil_field_lines_yz.png', dpi=300, bbox_inches='tight', facecolor='white')

# --- Bz-By plane (yz-plane, x=0): homogeneity ---
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
B = mp.getB(nmr_coil, grid)
B0 = mp.getB(nmr_coil, (0, 0, 0))
B0amp = np.linalg.norm(B0)
err = np.linalg.norm((B - B0) / B0amp, axis=2)

sp = ax.contourf(Y, Z, err * 100)
ax.plot([-radius_ssRF, radius_ssRF], [-width/2, -width/2], 'k-', lw=2, zorder=10)
ax.plot([-radius_ssRF, radius_ssRF], [width/2, width/2], 'k-', lw=2, zorder=10)
ax.plot([-radius_ssRF, -radius_ssRF], [-width/2, width/2], 'k-', lw=2, zorder=10, label='ssRF coil')
ax.plot([radius_ssRF, radius_ssRF], [-width/2, width/2], 'k-', lw=2, zorder=10)
ax.plot([-radius_NMR, radius_NMR], [0, 0], 'lime', lw=2, ls='--', zorder=11, label='NMR coil')
ax.legend(loc='upper right')

ax.set(
    title='Coil homogeneity error (Bz-By plane, x=0)',
    xlabel='y-position (m)',
    ylabel='z-position (m)',
    aspect='equal',
    xlim=(-0.015, 0.015),
    ylim=(-0.015, 0.015),
)
ax.margins(0)
plt.colorbar(sp, ax=ax, label='(% of B0)')
plt.tight_layout(pad=0.5)
# plt.show()
plt.savefig('coil_homogeneity_error_yz.png', dpi=300, bbox_inches='tight', facecolor='white')