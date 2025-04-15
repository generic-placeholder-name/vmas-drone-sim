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

def calculate_total_turn_angle(line: LineString) -> float:
    """
    Calculate total turning angle (in degrees) for a closed tour.
    Assumes the first and last coords in `line` are the same (closed).
    """
    # Get coords and drop any consecutive duplicates
    raw = list(line.coords)
    coords = [raw[0]]
    for p in raw[1:]:
        if p != coords[-1]:
            coords.append(p)
    # If not already closed, close it
    if coords[0] != coords[-1]:
        coords.append(coords[0])

    n = len(coords)
    if n < 4:  # need at least 3 distinct points + closure
        return 0.0

    total_turn = 0.0
    # For each vertex i, look at prev=i-1, curr=i, next=i+1 (mod n)
    for i in range(n):
        a = np.array(coords[(i-1) % n])
        b = np.array(coords[i])
        c = np.array(coords[(i+1) % n])

        v1 = a - b
        v2 = c - b
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            continue

        # interior angle
        cos_theta = np.dot(v1, v2) / (n1 * n2)
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

        # turning (exterior) angle = pi - interior
        turn = np.pi - theta
        total_turn += abs(np.degrees(turn))

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
        # This is a simple greedy approach, not the most efficient.s
        assert(len(final_seg) % 2 == 0), "Final segment should have an even number of points."
        for _ in range(num_iterations):
            for i in range(0, len(final_seg), 2):
                for j in range(i + 2, len(final_seg), 2): 
                    original_cost = (
                        dist[final_seg[(i - 1) % len(final_seg)]][final_seg[i]] + dist[final_seg[i + 1]][final_seg[(i + 2) % len(final_seg)]] +
                        dist[final_seg[(j - 1) % len(final_seg)]][final_seg[j]] + dist[final_seg[j + 1]][final_seg[(j + 2) % len(final_seg)]]
                    )

                    # Swap a-c and b-d
                    final_seg[i], final_seg[j] = final_seg[j], final_seg[i]
                    final_seg[i + 1], final_seg[j + 1] = final_seg[j + 1], final_seg[i + 1]
                    swap_ac_bd_cost = (
                        dist[final_seg[(i - 1) % len(final_seg)]][final_seg[i]] + dist[final_seg[i + 1]][final_seg[(i + 2) % len(final_seg)]] +
                        dist[final_seg[(j - 1) % len(final_seg)]][final_seg[j]] + dist[final_seg[j + 1]][final_seg[(j + 2) % len(final_seg)]]
                    )

                    if swap_ac_bd_cost > original_cost:
                        # We don't want to swap if it increases the cost, so revert.
                        # Swap a-c and b-d
                        final_seg[i], final_seg[j] = final_seg[j], final_seg[i]
                        final_seg[i + 1], final_seg[j + 1] = final_seg[j + 1], final_seg[i + 1]

                        # Swap a-d and b-c
                        final_seg[i + 1], final_seg[j] = final_seg[j], final_seg[i + 1]
                        final_seg[i], final_seg[j + 1] = final_seg[j + 1], final_seg[i]
                        swap_ad_bc_cost = (
                            dist[final_seg[(i - 1) % len(final_seg)]][final_seg[i]] + dist[final_seg[i + 1]][final_seg[(i + 2) % len(final_seg)]] +
                            dist[final_seg[(j - 1) % len(final_seg)]][final_seg[j]] + dist[final_seg[j + 1]][final_seg[(j + 2) % len(final_seg)]]
                        )
                        if swap_ad_bc_cost > original_cost:
                            # Revert to original
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
                    y_min, y_max = min(current_segment), max(current_segment)
                    segments.append(LineString([(x, y_min), (x, y_max)]))
                # Start new segment
                current_segment = [y]

        # Add final segment in column
        if current_segment:
            y_min, y_max = min(current_segment), max(current_segment)
            segments.append(LineString([(x, y_min), (x, y_max)]))

    return segments

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
    print("Waypoints:", waypoints)
    print("Segments:", segments)
    
    # Add base point connection
    segments.append(LineString([base_point, base_point]))
    
    # Merge all lines
    try:
        return merge_all_lines(segments, free_area, obstacles)
    except Exception as e:
        print(f"Path merging failed: {e}")
        return None

# === Parameters ===
box_coords = (0, 0, 245, 190)
obstacles = [
    (107, 38, 130, 68),
    (148, 86, 160, 116),
    (107, 7, 122, 22),
    (53, 129, 68, 144),
]
obs_tolerance = 3
survey_radius = 19

# Expand obstacles by tolerance
big_obstacles = [
    (x1 - obs_tolerance, y1 - obs_tolerance, x2 + obs_tolerance, y2 + obs_tolerance)
    for (x1, y1, x2, y2) in obstacles
]

# Generate waypoints over the whole area
waypoints, free_area = generate_grid_waypoints(
    box_coords, big_obstacles, grid_res=survey_radius*2
)

# Single drone base
base1 = (105, 150)

# Build vertical segments + base connection
segments = create_vertical_lines(waypoints, survey_radius, free_area)
segments.append(LineString([base1, base1]))  # ensures base is in the tour

# Merge into one closed tour
path1 = merge_all_lines(segments, free_area, big_obstacles)

# === Visualization ===
fig, ax = plt.subplots(figsize=(10, 8))

# Free area
x_free, y_free = free_area.exterior.xy
ax.fill(x_free, y_free, alpha=0.2, fc='green', ec='none')

# Original (un‑toleranced) obstacles in gray
for obs in obstacles:
    obs_poly = shapely_box(*obs)
    x_obs, y_obs = obs_poly.exterior.xy
    ax.fill(x_obs, y_obs, alpha=0.5, fc='gray', ec='gray')

# Waypoints
if waypoints:
    wp_x, wp_y = zip(*waypoints)
    ax.scatter(wp_x, wp_y, c='black', s=10, alpha=0.5, label='Waypoints')

# Single drone path
if path1:
    x1, y1 = path1.xy
    ax.plot(x1, y1, 'r-', lw=2, label='Drone Path')

# Metrics
length = path1.length if path1 else 0
turns  = calculate_total_turn_angle(path1) if path1 else 0
stats = [f"Length={length:.1f}", f"Turns={turns:.1f}°"]
ax.text(0.05, 0.95, "\n".join(stats),
        transform=ax.transAxes, verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.8))

# Final styling
ax.set_xlim(box_coords[0], box_coords[2])
ax.set_ylim(box_coords[1], box_coords[3])
ax.set_aspect('equal')
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.1))
plt.title("Single-Drone Survey Path")
plt.tight_layout()
plt.savefig(f"img/single_drone_survey_path_{survey_radius}.png")
plt.close()
