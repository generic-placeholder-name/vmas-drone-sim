import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point, box as shapely_box
from shapely.ops import unary_union
import networkx as nx

#########################
# Geometry and Pathfinding
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
# Sweep Segment Generation
#########################

def generate_sweep_segments(angle, radius, bounds, obstacles):
    """
    For a given angle, generate sweep segments as follows:
      - Create a family of parallel lines (spaced by 2*radius) covering the survey box.
      - Intersect each line with the free area (box minus obstacles).
      - Trim each resulting segment by removing a length equal to radius at each end.
        If a segment is too short, use its midpoint.
    Returns a list of segments as LineStrings.
    """
    free_area = create_free_area(bounds, obstacles)
    # Unit vector in sweep direction.
    d = np.array([np.cos(angle), np.sin(angle)])
    # Perpendicular unit vector (for offsets).
    n = np.array([-np.sin(angle), np.cos(angle)])
    
    # Determine offset range using box corners.
    corners = np.array([[box_coords[0], box_coords[1]],
                        [box_coords[0], box_coords[3]],
                        [box_coords[2], box_coords[1]],
                        [box_coords[2], box_coords[3]]])
    projections = corners.dot(n)
    c_min, c_max = projections.min(), projections.max()
    
    offsets = np.arange(c_min, c_max + 2*radius, 2*radius)
    segments = []
    far = max(box_coords[2]-box_coords[0], box_coords[3]-box_coords[1]) * 2
    
    for c in offsets:
        p0 = c * n
        p1 = p0 - far * d
        p2 = p0 + far * d
        line = LineString([tuple(p1), tuple(p2)])
        inter = free_area.intersection(line)
        if inter.is_empty:
            continue
        if inter.geom_type == 'MultiLineString':
            inter = list(inter.geoms)
        elif inter.geom_type == 'LineString':
            inter = [inter]
        else:
            continue
        
        for seg in inter:
            length = seg.length
            if length >= 2*radius:
                pt_start = np.array(seg.interpolate(radius).coords[0])
                pt_end   = np.array(seg.interpolate(length - radius).coords[0])
                trimmed = LineString([tuple(pt_start), tuple(pt_end)])
            else:
                mid = seg.interpolate(0.5, normalized=True)
                trimmed = LineString([mid, mid]) # Duplicating the point is a bit better.
            segments.append(trimmed)

    return segments, free_area

#########################
# Merging Lines by Connecting Closest Endpoints
#########################

def merge_all_lines(lines, free_area, obstacles, num_iterations=10):
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

#########################
# Main: Generate Sweep Segments and Merge Them
#########################

def generate_tour(box_coords, obstacles, base_point, survey_radius, angle):
    """
    Generate a full tour of the box (given by box_coords), avoiding obstacles, using back-and-forth heuristic.
    """
    # Generate sweep segments.
    segments, free_area = generate_sweep_segments(angle, survey_radius, box_coords, obstacles)
    segments.append(LineString([base_point, base_point]))
    # Merge all lines into one closed loop.
    try:
        full_closed_path = merge_all_lines(segments, free_area, obstacles)
    except ValueError as e:
        raise Exception(f"Error during merging: {e}")
    return full_closed_path

# Parameters.
box_coords = (0, 0, 800, 620)  # Survey area dimensions.
obstacles = [
    (350, 120, 426, 220),  
    (485, 280, 525, 378),   
    (350, 20, 400, 70),
    (175, 420, 225, 470),
]
survey_radius = 25           # Drone's survey radius.
base_point = (450, 200)  # You may choose a base point. (475, 200)

#########################
# Report and Visualization
#########################

for angle in range(0, 91, 10):
    try: 
        full_closed_path = generate_tour(box_coords, obstacles, base_point, survey_radius, np.radians(angle))
        total_length = full_closed_path.length
        total_turn_angle = calculate_total_turn_angle(full_closed_path)

        # Visualization.
        fig, ax = plt.subplots(figsize=(8,8))
        
        # Plot free area.
        x_free, y_free = shapely_box(*box_coords).exterior.xy
        ax.fill(x_free, y_free, alpha=0.2, fc='green', ec='green')
        
        # Plot obstacles.
        for obs in obstacles:
            obs_poly = shapely_box(*obs)
            x_obs, y_obs = obs_poly.exterior.xy
            ax.fill(x_obs, y_obs, alpha=0.5, fc='gray', ec='gray')
        
        # Plot the full closed path.
        x_path, y_path = full_closed_path.xy
        ax.plot(x_path, y_path, 'r--', lw=2, label='Merged Survey Path')
        
        ax.set_xlim(box_coords[0]-5, box_coords[2]+5)
        ax.set_ylim(box_coords[1]-5, box_coords[3]+5)
        ax.set_aspect('equal')
        ax.legend()
        plt.title(f"Angle: {angle:.1f}° | Total Path Length: {total_length:.1f} | Total Turn Angle: {total_turn_angle:.1f}°")
        plt.savefig(f"drone_merged_survey_path_{angle:.1f}.png")
        plt.close()
        
        print(f"Total path length: {total_length:.1f}")
        print(f"Total turning angle: {total_turn_angle:.1f}°")
        print("Plot saved as 'drone_merged_survey_path.png'. Open it to view the survey path.")
    except Exception as e:
        print("Exception encountered:", e)
