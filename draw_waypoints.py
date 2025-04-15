import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, box as shapely_box
from shapely.ops import unary_union

def create_free_area(box_coords, obstacles):
    """Create the free area by subtracting obstacles from survey box."""
    free_area = shapely_box(*box_coords)
    obs_polys = [shapely_box(*obs) for obs in obstacles]
    if obs_polys:
        obstacles_union = unary_union(obs_polys)
        free_area = free_area.difference(obstacles_union)
    return free_area

def generate_grid_waypoints(box_coords, obstacles, grid_res):
    """Generate grid waypoints within free area"""
    world_width = box_coords[2] - box_coords[0]
    world_height = box_coords[3] - box_coords[1]
    
    waypoints = []
    free_area = create_free_area(box_coords, obstacles)
    
    # Generate grid points with collision checking
    for x in np.arange(grid_res/2, world_width, grid_res):
        for y in np.arange(grid_res/2, world_height, grid_res):
            pt = Point(x + box_coords[0], y + box_coords[1])
            if free_area.contains(pt):
                waypoints.append((pt.x, pt.y))
    
    return waypoints, free_area

# Parameters
box_coords = (0, 0, 245, 190)
obstacles = [
    (107, 38, 130, 68),
    (148, 86, 160, 116),
    (107, 7, 122, 22),
    (53, 129, 68, 144),
]
base1 = (105, 150)
base2 = (125, 150)
survey_radius = 19
grid_res = survey_radius * 2

# Generate waypoints
waypoints, free_area = generate_grid_waypoints(box_coords, obstacles, grid_res)

# Visualization
fig, ax = plt.subplots(figsize=(10, 8))

# Plot free area
x_free, y_free = free_area.exterior.xy
ax.fill(x_free, y_free, alpha=0.2, fc='green', ec='none')

# Plot obstacles
for obs in obstacles:
    obs_poly = shapely_box(*obs)
    x_obs, y_obs = obs_poly.exterior.xy
    ax.fill(x_obs, y_obs, alpha=0.5, fc='red', ec='red')

# Plot waypoints
if waypoints:
    wp_x, wp_y = zip(*waypoints)
    ax.scatter(wp_x, wp_y, c='blue', s=15, alpha=0.7, label=f'Waypoints ({len(waypoints)} points)')

ax.plot(*base1, 'ro', label='Base 1')
ax.plot(*base2, 'bo', label='Base 2')

# Plot settings
ax.set_xlim(box_coords[0], box_coords[2])
ax.set_ylim(box_coords[1], box_coords[3])
ax.set_aspect('equal')
ax.legend()
plt.title(f"Survey Waypoints (Grid Resolution: {grid_res:.1f}m)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("img/survey_waypoints.png", dpi=300)
plt.close()

print(f"Generated {len(waypoints)} waypoints")
print(f"First 5 waypoints: {waypoints[:5]}")