from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point, box as shapely_box
from shapely.ops import unary_union
import networkx as nx

#########################
# Geometry and Pathfinding (No changes here)
#########################

def create_free_area(box_coords, obstacles):
    """Create the free area (as a Polygon) by subtracting obstacle boxes from the survey box."""
    free_area = shapely_box(*box_coords)
    obs_polys = [shapely_box(*obs) for obs in obstacles]
    if obs_polys:
        obstacles_union = unary_union(obs_polys)
        free_area = free_area.difference(obstacles_union)
    return free_area

def build_visibility_graph(p1, p2, free_area, obstacles):
    """
    Build a visibility graph of key nodes:
      - p1 and p2,
      - vertices of free_area boundary,
      - vertices of obstacles (boxes).
    Connect two nodes if the straight line between them is completely inside free_area.
    Returns a NetworkX graph.
    """
    nodes = set()
    nodes.add(p1)
    nodes.add(p2)
    # Add free area boundary vertices.
    for coord in free_area.exterior.coords:
        nodes.add(coord)
    # Add obstacle vertices.
    for obs in obstacles:
        obs_poly = shapely_box(*obs)
        for coord in obs_poly.exterior.coords:
            if free_area.covers(Point(coord)):
                nodes.add(coord)
    nodes = list(nodes)
    
    G = nx.Graph()
    for node in nodes:
        G.add_node(node)
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            a = nodes[i]
            b = nodes[j]
            line = LineString([a, b])
            if free_area.covers(line):
                dist = np.linalg.norm(np.array(a) - np.array(b))
                G.add_edge(a, b, weight=dist)
                G.add_edge(b, a, weight=dist)
    return G

def find_path(p1, p2, free_area, obstacles):
    """Return a collision-free path (list of coordinates) from p1 to p2."""
    G = build_visibility_graph(p1, p2, free_area, obstacles)
    try:
        path = nx.shortest_path(G, source=p1, target=p2, weight='weight')
    except nx.NetworkXNoPath:
        raise ValueError(f"No collision-free path found between {p1} and {p2}.")
    return path

def calculate_total_turn_angle(path):
    """Calculate the total turning angle (in degrees) along a path (LineString)."""
    coords = list(path.coords)
    total_turn = 0
    for i in range(1, len(coords)-1):
        a = np.array(coords[i-1])
        b = np.array(coords[i])
        c = np.array(coords[i+1])
        v1 = a - b
        v2 = c - b
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            continue
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        total_turn += np.degrees(angle)
    return total_turn

#########################
# Merging Lines by Connecting Closest Endpoints (No changes here)
#########################

def merge_all_lines(lines, free_area, obstacles, num_iterations=50):
    # Precompute segments and points from input lines.
    segments_orig = []
    points = []
    for line in lines:
        x, y = list(line.coords)
        segments_orig.append([len(points), len(points) + 1])
        points.extend((x, y))

    num_points = len(points)
    
    # Precompute collision-free paths and distances between all pairs of points.
    path = [[None] * num_points for _ in range(num_points)]
    dist = [[0] * num_points for _ in range(num_points)]
    for i in range(num_points):
        for j in range(i):
            # Compute the collision-free path between points[i] and points[j]
            path[i][j] = find_path(points[i], points[j], free_area, obstacles)
            path[j][i] = path[i][j][::-1]
            seg_length = sum(
                np.linalg.norm(np.array(path[i][j][k+1]) - np.array(path[i][j][k]))
                for k in range(len(path[i][j]) - 1)
            )
            dist[i][j] = seg_length
            dist[j][i] = seg_length

    best_final_length = np.inf
    best_full_path_coords = None

    # Run multiple iterations, reusing the precomputed data
    for _ in range(num_iterations):
        # Work on a copy of the original segments list.
        segments = [seg.copy() for seg in segments_orig]
        
        # Repeatedly connect the two segments with the smallest (perturbed) distance.
        while len(segments) > 1:
            best_length = np.inf
            best_i = None
            best_j = None
            best_pt_i = None
            best_pt_j = None

            # Consider every unordered pair of segments.
            for i in range(len(segments)):
                for j in range(i+1, len(segments)):
                    for pt_i in (segments[i][0], segments[i][-1]):
                        for pt_j in (segments[j][0], segments[j][-1]):
                            candidate = dist[pt_i][pt_j] + np.random.uniform(-1e-9, 1e-9)
                            if candidate < best_length:
                                best_length = candidate
                                best_i = i
                                best_j = j
                                best_pt_i = pt_i
                                best_pt_j = pt_j

            if best_i is None:
                raise ValueError("No valid connection found between any two lines.")

            # Reverse segments if necessary so that endpoints match.
            if best_pt_i == segments[best_i][0]:
                segments[best_i].reverse()
            if best_pt_j == segments[best_j][-1]:
                segments[best_j].reverse()
            first_seg = segments[best_i]
            second_seg = segments[best_j]

            # Merge the two segments.
            segments.append(first_seg + second_seg)
            segments.remove(first_seg)
            segments.remove(second_seg)

        # Only one merged segment remains.
        final_seg = segments[0]

        # Do 2-opt optimization on the merged segment. 
        # Since the tour is made up of 2-point segments, we do 2-opt on 2 points at a time.
        # This is a simple greedy approach, not the most efficient.
        assert(len(final_seg) % 2 == 0), "Final segment should have an even number of points."
        for i in range(0, len(final_seg), 2):
            for j in range(i + 4, len(final_seg), 2): # The implementation has problems with consecutive segments, so we just skip them.
                # Check if swapping improves the path.
                a = final_seg[i]
                b = final_seg[i + 1]
                c = final_seg[j]
                d = final_seg[j + 1]
                original_cost = (
                    dist[final_seg[(i - 1) % len(final_seg)]][a] + dist[b][final_seg[(i + 2) % len(final_seg)]] +
                    dist[final_seg[(j - 1) % len(final_seg)]][c] + dist[d][final_seg[(j + 2) % len(final_seg)]]
                )
                swap_ac_bd_cost = (
                    dist[final_seg[(i - 1) % len(final_seg)]][c] + dist[d][final_seg[(i + 2) % len(final_seg)]] +
                    dist[final_seg[(j - 1) % len(final_seg)]][a] + dist[b][final_seg[(j + 2) % len(final_seg)]]
                )
                swap_ad_bc_cost = (
                    dist[final_seg[(i - 1) % len(final_seg)]][d] + dist[c][final_seg[(i + 2) % len(final_seg)]] +
                    dist[final_seg[(j - 1) % len(final_seg)]][b] + dist[a][final_seg[(j + 2) % len(final_seg)]]
                )

                if swap_ac_bd_cost < original_cost and swap_ac_bd_cost < swap_ad_bc_cost:
                    # Swap a-c and b-d
                    final_seg[i], final_seg[j] = final_seg[j], final_seg[i]
                    final_seg[i + 1], final_seg[j + 1] = final_seg[j + 1], final_seg[i + 1]
                elif swap_ad_bc_cost < original_cost:
                    # Swap a-d and b-c
                    final_seg[i + 1], final_seg[j] = final_seg[j], final_seg[i + 1]
                    final_seg[i], final_seg[j + 1] = final_seg[j + 1], final_seg[i]

        full_path_coords = []
        for i in range(len(final_seg)):
            cur = final_seg[i]
            nxt = final_seg[(i + 1) % len(final_seg)]
            full_path_coords.append(points[cur])
            # Append intermediate points from the precomputed path (excluding endpoints).
            full_path_coords.extend(path[cur][nxt][1:-1])
        full_path_coords.append(points[final_seg[0]])

        # Compute the total length of the closed path.
        final_length = 0
        for i in range(len(full_path_coords) - 1):
            final_length += np.linalg.norm(np.array(full_path_coords[i+1]) - np.array(full_path_coords[i]))

        # Save the best final iteration.
        if final_length < best_final_length:
            best_final_length = final_length
            best_full_path_coords = full_path_coords

    return LineString(best_full_path_coords)

def create_vertical_lines(waypoints, survey_radius, free_area):
    """Create obstacle-aware vertical segments through waypoint columns"""
    # Group waypoints by x-coordinate
    columns = defaultdict(list)
    for x, y in waypoints:
        columns[x].append(y)

    segments = []
    
    # Process each column
    for x in sorted(columns.keys()):
        ys = sorted(columns[x])
        if not ys:
            continue

        current_segment = []
        
        # Scan through sorted y-values
        for y in ys:
            if not current_segment:
                # Start new segment
                current_segment.append(y)
                continue
                
            # Check line from last point to current point
            last_y = current_segment[-1]
            test_line = LineString([(x, last_y), (x, y)])
            
            if free_area.covers(test_line):
                # Continue current segment
                current_segment.append(y)
            else:
                # Finalize current segment
                if len(current_segment) >= 1:
                    _finalize_segment(x, current_segment, segments, 
                                         survey_radius, free_area)
                # Start new segment
                current_segment = [y]

        # Add final segment in column
        if current_segment:
            _finalize_segment(x, current_segment, segments,
                                 survey_radius, free_area)

    return segments

def _finalize_segment(x, y_values, segments, radius, free_area):
    """Create and trim a vertical segment from collected y-values"""
    y_min = min(y_values) - radius
    y_max = max(y_values) + radius
    line = LineString([(x, y_min), (x, y_max)])
    
    # Intersect with free area
    inter = free_area.intersection(line)
    if inter.is_empty:
        return
    
    # Handle different geometry types
    if inter.geom_type == 'MultiLineString':
        parts = list(inter.geoms)
    else:
        parts = [inter]

    # Trim each valid segment
    for part in parts:
        length = part.length
        if length >= 2*radius:
            start = part.interpolate(radius)
            end = part.interpolate(length - radius)
            segments.append(LineString([start, end]))
        else:
            mid = part.interpolate(0.5)
            segments.append(LineString([mid, mid]))

# Modified grid generation with obstacle checking
def generate_grid_waypoints(box_coords, obstacles, grid_res):
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

def generate_drone_path(waypoints, base_point, survey_radius, free_area, obstacles):
    """Generate path for single drone"""
    # Create vertical lines through waypoints
    segments = create_vertical_lines(waypoints, survey_radius, free_area)
    
    # Add base point connection
    segments.append(LineString([base_point, base_point]))
    
    # Merge all lines
    try:
        return merge_all_lines(segments, free_area, obstacles)
    except Exception as e:
        print(f"Path merging failed: {e}")
        return None

# Split survey area
box_coords = (0, 0, 800, 620)
split_x = 460
left_box = (0, 0, split_x, 620)
right_box = (split_x, 0, 800, 620)

obstacles = [
    (350, 120, 426, 220),
    (485, 280, 525, 378),
    (350, 20, 400, 70),
    (175, 420, 225, 470),
]
obs_tolerance = 5
survey_radius = 25 # 620 // 5

big_obstacles = [
    (x1 - obs_tolerance, y1 - obs_tolerance, x2 + obs_tolerance, y2 + obs_tolerance)
    for (x1, y1, x2, y2) in obstacles
]

# Generate waypoints
waypoints, free_area = generate_grid_waypoints(box_coords, big_obstacles, survey_radius*2)

# Split waypoints between drones
left_waypoints = [p for p in waypoints if p[0] <= split_x]
right_waypoints = [p for p in waypoints if p[0] > split_x]

# Generate paths
base1 = (450, 200)
base2 = (475, 200)
path1 = generate_drone_path(left_waypoints, base1, survey_radius, free_area, big_obstacles)
path2 = generate_drone_path(right_waypoints, base2, survey_radius, free_area, big_obstacles)

# Visualization
fig, ax = plt.subplots(figsize=(10, 8))

# Plot free area and obstacles
x_free, y_free = free_area.exterior.xy
ax.fill(x_free, y_free, alpha=0.2, fc='green', ec='none')

for obs in obstacles:
    obs_poly = shapely_box(*obs)
    x_obs, y_obs = obs_poly.exterior.xy
    ax.fill(x_obs, y_obs, alpha=0.5, fc='gray', ec='gray')

# Plot waypoints
wp_x, wp_y = zip(*waypoints) if waypoints else ([], [])
ax.scatter(wp_x, wp_y, c='black', s=10, alpha=0.5, label='Waypoints')

# Plot drone paths
if path1:
    x1, y1 = path1.xy
    ax.plot(x1, y1, 'r-', lw=2, label='Drone 1 Path')
if path2:
    x2, y2 = path2.xy
    ax.plot(x2, y2, 'b-', lw=2, label='Drone 2 Path')

# Move legend to the bottom of the graph
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=3)

# Add metrics
stats = []
if path1:
    stats.append(f"Drone 1: Length={path1.length:.1f}, Turns={calculate_total_turn_angle(path1):.1f}°")
if path2:
    stats.append(f"Drone 2: Length={path2.length:.1f}, Turns={calculate_total_turn_angle(path2):.1f}°")
    
if stats:
    ax.text(0.05, 0.95, "\n".join(stats), transform=ax.transAxes,
            verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

ax.set_xlim(box_coords[0], box_coords[2])
ax.set_ylim(box_coords[1], box_coords[3])
ax.set_aspect('equal')
ax.legend()

plt.title("Waypoint-Based Survey Path")
plt.savefig(f"two_drones_survey_path_waypoints_{survey_radius}.png")
plt.close()