import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
plt.ioff()

def objective_function(x, y):
    """
    A function with multiple local minima.
    We want to find the maximum among these minima.
    """
    z = np.sin(x) * np.cos(y) + 0.5 * np.sin(2 * x) * np.cos(2 * y) + 0.3 * np.sin(3 * x + 1) * np.cos(3 * y - 1) + 0.25 * np.sin(4 * x) * np.sin(2 * y) + 0.2 * np.cos(5 * x - 2 * y) + 0.15 * np.sin(x * y * 0.5) + 0.3 * x ** 2 + 0.3 * y ** 2 - 0.1 * (x ** 2 + y ** 2) ** 0.5 + 0.1 * np.sin(7 * x) * np.cos(4 * y)
    return -z

def gradient(x, y, h=1e-05):
    """
    Compute numerical gradient of the objective function.
    """
    dx = (objective_function(x + h, y) - objective_function(x - h, y)) / (2 * h)
    dy = (objective_function(x, y + h) - objective_function(x, y - h)) / (2 * h)
    return (dx, dy)
learning_rate_initial = 0.15
learning_rate_decay = 0.99
num_iterations = 100
momentum = 0.8
x_start = np.random.uniform(-2, 2)
y_start = np.random.uniform(-2, 2)
trajectory_x = [x_start]
trajectory_y = [y_start]
trajectory_z = [objective_function(x_start, y_start)]
(vx, vy) = (0, 0)
(x, y) = (x_start, y_start)
for i in range(num_iterations):
    (dx, dy) = gradient(x, y)
    noise_scale = 0.1 * 0.95 ** i
    dx += np.random.normal(0.3, noise_scale)
    dy += np.random.normal(1.34, noise_scale)
    learning_rate = learning_rate_initial * learning_rate_decay ** i
    vx = momentum * vx - learning_rate * dx
    vy = momentum * vy - learning_rate * dy
    x += vx
    y += vy
    x = np.clip(x, -3, 3)
    y = np.clip(y, -3, 3)
    trajectory_x.append(x)
    trajectory_y.append(y)
    trajectory_z.append(objective_function(x, y))
x_mesh = np.linspace(-3, 3, 200)
y_mesh = np.linspace(-3, 3, 200)
(X, Y) = np.meshgrid(x_mesh, y_mesh)
Z = objective_function(X, Y)
fig = plt.figure(figsize=(16, 6))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1])
ax1 = fig.add_subplot(gs[0], projection='3d')
ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, edgecolor='none')
ax1.set_xlabel('X', fontsize=10)
ax1.set_ylabel('Y', fontsize=10)
ax1.set_zlabel('f(X, Y)', fontsize=10)
ax1.set_title('3D Surface - SGD Trajectory', fontsize=12, fontweight='bold')
ax1.view_init(elev=20, azim=45)
(trajectory_line_3d,) = ax1.plot([], [], [], 'r-', linewidth=2, alpha=0.7, label='Path')
(current_point_3d,) = ax1.plot([], [], [], 'ro', markersize=10, label='Current')
(start_point_3d,) = ax1.plot([x_start], [y_start], [trajectory_z[0]], 'go', markersize=8, label='Start', zorder=10)
ax2 = fig.add_subplot(gs[1])
contour = ax2.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.8)
ax2.contour(X, Y, Z, levels=50, colors='black', alpha=0.3, linewidths=0.5)
ax2.set_xlabel('X', fontsize=10)
ax2.set_ylabel('Y', fontsize=10)
ax2.set_title('2D Contour - Top View', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
plt.colorbar(contour, ax=ax2, label='f(X, Y)')
(trajectory_line_2d,) = ax2.plot([], [], 'r-', linewidth=2, alpha=0.7, label='Path')
(current_point_2d,) = ax2.plot([], [], 'ro', markersize=10, label='Current')
start_point_2d = ax2.plot([x_start], [y_start], 'go', markersize=8, label='Start', zorder=10)
arrow_2d = None
ax3 = fig.add_subplot(gs[2])
ax3.set_xlabel('Iteration', fontsize=10)
ax3.set_ylabel('Objective Value', fontsize=10)
ax3.set_title('Convergence Plot', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, num_iterations)
ax3.set_ylim(min(trajectory_z) * 1.1, max(trajectory_z) * 1.1)
(loss_line,) = ax3.plot([], [], 'b-', linewidth=2, label='Objective')
(current_loss_point,) = ax3.plot([], [], 'ro', markersize=8)
ax3.axhline(y=min(trajectory_z), color='g', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Best: {min(trajectory_z):.4f}')
ax3.legend(fontsize=9)
info_text = ax1.text2D(0.02, 0.98, '', transform=ax1.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

def init():
    """Initialize animation."""
    trajectory_line_3d.set_data([], [])
    trajectory_line_3d.set_3d_properties([])
    current_point_3d.set_data([], [])
    current_point_3d.set_3d_properties([])
    trajectory_line_2d.set_data([], [])
    current_point_2d.set_data([], [])
    loss_line.set_data([], [])
    current_loss_point.set_data([], [])
    return (trajectory_line_3d, current_point_3d, trajectory_line_2d, current_point_2d, loss_line, current_loss_point, info_text)

def animate(frame):
    """Animation function."""
    global arrow_2d
    traj_x = trajectory_x[:frame + 1]
    traj_y = trajectory_y[:frame + 1]
    traj_z = trajectory_z[:frame + 1]
    trajectory_line_3d.set_data(traj_x, traj_y)
    trajectory_line_3d.set_3d_properties(traj_z)
    if frame > 0:
        current_point_3d.set_data([traj_x[-1]], [traj_y[-1]])
        current_point_3d.set_3d_properties([traj_z[-1]])
    trajectory_line_2d.set_data(traj_x, traj_y)
    if frame > 0:
        current_point_2d.set_data([traj_x[-1]], [traj_y[-1]])
    if arrow_2d:
        arrow_2d.remove()
        arrow_2d = None
    if frame > 0 and frame < len(trajectory_x) - 1:
        (dx, dy) = gradient(traj_x[-1], traj_y[-1])
        scale = 0.3
        arrow_2d = ax2.arrow(traj_x[-1], traj_y[-1], -dx * scale, -dy * scale, head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.7, linewidth=2)
    iterations = list(range(len(traj_z)))
    loss_line.set_data(iterations, traj_z)
    if frame > 0:
        current_loss_point.set_data([frame], [traj_z[-1]])
    if frame > 0:
        learning_rate = learning_rate_initial * learning_rate_decay ** frame
        info_text.set_text(f'Iteration: {frame}/{num_iterations - 1}\nPosition: ({traj_x[-1]:.3f}, {traj_y[-1]:.3f})\nValue: {traj_z[-1]:.4f}\nLearning Rate: {learning_rate:.4f}')
    ax1.view_init(elev=20, azim=45 + frame * 0.3)
    return (trajectory_line_3d, current_point_3d, trajectory_line_2d, current_point_2d, loss_line, current_loss_point, info_text)
print('Creating animation...')
anim = FuncAnimation(fig, animate, init_func=init, frames=len(trajectory_x), interval=200, blit=False, repeat=True)
ax1.legend(loc='upper right', fontsize=9)
ax2.legend(loc='upper right', fontsize=9)
plt.tight_layout()
print("Saving animation as 'sgd_animation.gif'...")
anim.save('sgd_animation.gif', writer='pillow', fps=10, dpi=80)
print('Animation saved successfully!')
plt.close(fig)
print('\nCreating simplified 2D path animation...')
fig_simple = plt.figure(figsize=(10, 8))
ax_simple = fig_simple.add_subplot(111)
contour_simple = ax_simple.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.85)
ax_simple.contour(X, Y, Z, levels=50, colors='black', alpha=0.25, linewidths=0.5)
ax_simple.set_xlabel('X', fontsize=14, fontweight='bold')
ax_simple.set_ylabel('Y', fontsize=14, fontweight='bold')
ax_simple.set_title('Stochastic Gradient Descent - Finding Maximum Minimum', fontsize=16, fontweight='bold', pad=20)
ax_simple.grid(True, alpha=0.2, linestyle='--')
cbar = plt.colorbar(contour_simple, ax=ax_simple, label='Objective Value')
cbar.ax.tick_params(labelsize=11)
(trajectory_line_simple,) = ax_simple.plot([], [], 'r-', linewidth=3, alpha=0.8, label='SGD Path', zorder=5)
(current_point_simple,) = ax_simple.plot([], [], 'ro', markersize=14, label='Current Position', zorder=10)
start_marker = ax_simple.plot([x_start], [y_start], 'go', markersize=12, label='Start', zorder=8)
gradient_arrow_simple = None
info_box = ax_simple.text(0.02, 0.98, '', transform=ax_simple.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.8', facecolor='white', edgecolor='black', alpha=0.9, linewidth=2))
ax_simple.legend(loc='upper right', fontsize=12, framealpha=0.9)

def init_simple():
    """Initialize simple animation."""
    trajectory_line_simple.set_data([], [])
    current_point_simple.set_data([], [])
    return (trajectory_line_simple, current_point_simple, info_box)

def animate_simple(frame):
    """Animation function for simple version."""
    global gradient_arrow_simple
    traj_x = trajectory_x[:frame + 1]
    traj_y = trajectory_y[:frame + 1]
    traj_z = trajectory_z[:frame + 1]
    trajectory_line_simple.set_data(traj_x, traj_y)
    if frame > 0:
        current_point_simple.set_data([traj_x[-1]], [traj_y[-1]])
    if gradient_arrow_simple:
        gradient_arrow_simple.remove()
        gradient_arrow_simple = None
    if frame > 0 and frame < len(trajectory_x) - 1:
        (dx, dy) = gradient(traj_x[-1], traj_y[-1])
        scale = 0.4
        gradient_arrow_simple = ax_simple.arrow(traj_x[-1], traj_y[-1], -dx * scale, -dy * scale, head_width=0.15, head_length=0.12, fc='yellow', ec='orange', alpha=0.9, linewidth=2.5, zorder=9, length_includes_head=True)
    if frame > 0:
        learning_rate = learning_rate_initial * learning_rate_decay ** frame
        improvement = trajectory_z[0] - traj_z[-1]
        info_box.set_text(f'Iteration: {frame:3d} / {num_iterations - 1}\nPosition: ({traj_x[-1]:6.3f}, {traj_y[-1]:6.3f})\nObjective: {traj_z[-1]:7.4f}\nBest So Far: {min(traj_z):7.4f}\nImprovement: {improvement:7.4f}\nLearning Rate: {learning_rate:.4f}')
    else:
        info_box.set_text(f'Iteration: {frame:3d} / {num_iterations - 1}\nStarting optimization...')
    return (trajectory_line_simple, current_point_simple, info_box)
anim_simple = FuncAnimation(fig_simple, animate_simple, init_func=init_simple, frames=len(trajectory_x), interval=120, blit=False, repeat=True)
plt.tight_layout()
print("Saving simplified SGD path animation as 'sgd_path_simple.gif'...")
anim_simple.save('sgd_path_simple.gif', writer='pillow', fps=8, dpi=100)
print('Simplified animation saved successfully!')
plt.close(fig_simple)
(fig_static, axes) = plt.subplots(1, 2, figsize=(14, 6))
ax_3d = fig_static.add_subplot(121, projection='3d')
ax_3d.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5, edgecolor='none')
ax_3d.plot(trajectory_x, trajectory_y, trajectory_z, 'r-', linewidth=2.5, alpha=0.8, label='SGD Path')
ax_3d.plot([x_start], [y_start], [trajectory_z[0]], 'go', markersize=10, label='Start')
ax_3d.plot([trajectory_x[-1]], [trajectory_y[-1]], [trajectory_z[-1]], 'ro', markersize=10, label='End')
ax_3d.set_xlabel('X', fontsize=11)
ax_3d.set_ylabel('Y', fontsize=11)
ax_3d.set_zlabel('f(X, Y)', fontsize=11)
ax_3d.set_title('Complete SGD Trajectory (3D)', fontsize=13, fontweight='bold')
ax_3d.legend(fontsize=10)
ax_3d.view_init(elev=20, azim=45)
ax_2d = axes[1]
contour_static = ax_2d.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.8)
ax_2d.contour(X, Y, Z, levels=15, colors='black', alpha=0.3, linewidths=0.5)
ax_2d.plot(trajectory_x, trajectory_y, 'r-', linewidth=2.5, alpha=0.8, label='SGD Path')
ax_2d.plot([x_start], [y_start], 'go', markersize=10, label='Start', zorder=10)
ax_2d.plot([trajectory_x[-1]], [trajectory_y[-1]], 'ro', markersize=10, label='End', zorder=10)
ax_2d.set_xlabel('X', fontsize=11)
ax_2d.set_ylabel('Y', fontsize=11)
ax_2d.set_title('Complete SGD Trajectory (2D)', fontsize=13, fontweight='bold')
ax_2d.grid(True, alpha=0.3)
ax_2d.legend(fontsize=10)
plt.colorbar(contour_static, ax=ax_2d, label='f(X, Y)')
plt.tight_layout()
plt.savefig('sgd_trajectory_static.png', dpi=300, bbox_inches='tight')
print("Static plot saved as 'sgd_trajectory_static.png'")
plt.close('all')
print('Creating standalone 3D trajectory plot...')
fig_3d_standalone = plt.figure(figsize=(12, 10))
ax_3d_standalone = fig_3d_standalone.add_subplot(111, projection='3d')
ax_3d_standalone.plot_surface(X, Y, Z, cmap='plasma', alpha=0.7, edgecolor='none', antialiased=True, linewidth=0, rcount=150, ccount=150)
ax_3d_standalone.plot(trajectory_x, trajectory_y, trajectory_z, 'r-', linewidth=3.5, alpha=0.95, label='SGD Path', zorder=20)
ax_3d_standalone.plot([x_start], [y_start], [trajectory_z[0]], 'go', markersize=14, label='Start', zorder=25, markeredgecolor='darkgreen', markeredgewidth=2)
ax_3d_standalone.plot([trajectory_x[-1]], [trajectory_y[-1]], [trajectory_z[-1]], 'ro', markersize=14, label='End', zorder=25, markeredgecolor='darkred', markeredgewidth=2)
num_markers = 15
marker_indices = np.linspace(0, len(trajectory_x) - 1, num_markers, dtype=int)
ax_3d_standalone.scatter([trajectory_x[i] for i in marker_indices], [trajectory_y[i] for i in marker_indices], [trajectory_z[i] for i in marker_indices], c='yellow', s=50, alpha=0.8, edgecolors='orange', linewidth=1.5, zorder=22, depthshade=True)
ax_3d_standalone.set_xlabel('X', fontsize=14, fontweight='bold', labelpad=10)
ax_3d_standalone.set_ylabel('Y', fontsize=14, fontweight='bold', labelpad=10)
ax_3d_standalone.set_zlabel('f(X, Y)', fontsize=14, fontweight='bold', labelpad=10)
ax_3d_standalone.set_title('Stochastic Gradient Descent - 3D Trajectory on Chaotic Surface', fontsize=16, fontweight='bold', pad=20)
ax_3d_standalone.view_init(elev=25, azim=135)
ax_3d_standalone.legend(fontsize=12, loc='upper left', framealpha=0.95)
ax_3d_standalone.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax_3d_standalone.set_xlim(-3, 3)
ax_3d_standalone.set_ylim(-3, 3)
info_str = f'Iterations: {num_iterations}\nStart: ({x_start:.3f}, {y_start:.3f})\nEnd: ({trajectory_x[-1]:.3f}, {trajectory_y[-1]:.3f})\nBest Value: {min(trajectory_z):.4f}\nImprovement: {trajectory_z[0] - min(trajectory_z):.4f}'
ax_3d_standalone.text2D(0.02, 0.98, info_str, transform=ax_3d_standalone.transAxes, fontsize=11, verticalalignment='top', family='monospace', bbox=dict(boxstyle='round,pad=0.8', facecolor='white', edgecolor='black', alpha=0.9, linewidth=1.5))
plt.tight_layout()
plt.savefig('sgd_trajectory_3d_standalone.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print("Standalone 3D plot saved as 'sgd_trajectory_3d_standalone.png'")
print('Creating rotating 3D trajectory animation...')

def animate_3d_rotation(frame):
    """Rotate the 3D view."""
    ax_3d_standalone.view_init(elev=25, azim=frame * 2)
    return (ax_3d_standalone,)
anim_3d_rotating = FuncAnimation(fig_3d_standalone, animate_3d_rotation, frames=180, interval=50, blit=False, repeat=True)
print("Saving rotating 3D trajectory as 'sgd_trajectory_3d_rotating.gif'...")
anim_3d_rotating.save('sgd_trajectory_3d_rotating.gif', writer='pillow', fps=20, dpi=100)
print('Rotating 3D animation saved successfully!')
plt.close(fig_3d_standalone)
print(f"\n{'=' * 60}")
print(f'Final Result:')
print(f'  Starting position: ({x_start:.4f}, {y_start:.4f})')
print(f'  Starting value: {trajectory_z[0]:.4f}')
print(f'  Final position: ({trajectory_x[-1]:.4f}, {trajectory_y[-1]:.4f})')
print(f'  Final value: {trajectory_z[-1]:.4f}')
print(f'  Best value found: {min(trajectory_z):.4f}')
print(f'  Improvement: {trajectory_z[0] - min(trajectory_z):.4f}')
