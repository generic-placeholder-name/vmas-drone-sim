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

def merge_all_lines(lines, free_area, obstacles):
    """
    Given a list of line dictionaries, repeatedly merge two lines by connecting the two closest endpoints
    (using pathfinding that avoids obstacles). At each iteration:
      - For every pair of lines, for each combination of one endpoint from one and one from the other,
        compute the collision-free connection using find_path.
      - Choose the pair (and corresponding endpoints) with the smallest path length.
      - Merge the two lines into one.
    Continue until only one line remains.
    Then, finally, connect the two endpoints of that line to close the loop (using pathfinding).
    Returns the complete closed path as a LineString.
    """
    segments = []
    points = []
    for line in lines: 
        x, y = list(line.coords)
        segments.append([len(points), len(points) + 1])
        points.extend((x, y))
    
    path = [[None] * len(points) for _ in range(len(points))]
    dist = [[0] * len(points) for _ in range(len(points))]

    for i in range(len(points)):
        for j in range(i):
            path[i][j] = find_path(points[i], points[j], free_area, obstacles)
            path[j][i] = path[i][j][::-1]
            dist[i][j] = sum(np.linalg.norm(np.array(path[i][j][k+1]) - np.array(path[i][j][k])) for k in range(len(path[i][j])-1))
            dist[j][i] = dist[i][j]

    while len(segments) > 1:
        best_length = np.inf
        best_i = None
        best_j = None
        best_pt_i = None
        best_pt_j = None
        
        # For each unordered pair of lines:
        for i in range(len(segments)):
            for j in range(i+1, len(segments)):
                for pt_i in (segments[i][0], segments[i][-1]):
                    for pt_j in (segments[j][0], segments[j][-1]):
                        try:
                            candidate = dist[pt_i][pt_j]
                        except ValueError:
                            continue
                        if candidate < best_length:
                            best_length = candidate
                            best_i = i
                            best_j = j
                            best_pt_i = pt_i
                            best_pt_j = pt_j
        if best_i is None:
            raise ValueError("No valid connection found between any two lines.")

        # Merge two lines
        if best_pt_i == segments[best_i][0]:
            segments[best_i].reverse()
        if best_pt_j == segments[best_j][-1]:
            segments[best_j].reverse()
        first_seg, second_seg = segments[best_i], segments[best_j]

        # Append merged_line to list.
        segments.append(first_seg + second_seg)
        segments.remove(first_seg)
        segments.remove(second_seg)
    
    # At this point, only one line remains.
    final_seg = segments[0]
    full_path_coords = []
    for i in range(len(final_seg)):
        cur, nxt = final_seg[i], final_seg[(i + 1) % len(final_seg)]
        full_path_coords.append(points[cur])
        full_path_coords.extend(path[cur][nxt][1:-1])
    full_path_coords.append(points[final_seg[0]])

    return LineString(full_path_coords)

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
box_coords = (0, 0, 100, 100)  # Survey area dimensions.
obstacles = [
    (30, 30, 50, 50),  # A square obstacle.
    (60, 10, 80, 20)   # A rectangular obstacle.
]
survey_radius = 5           # Drone's survey radius.
base_point = (25, 0)  # You may choose a base point.

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
