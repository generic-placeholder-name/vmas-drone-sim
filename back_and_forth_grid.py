import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, box as shapely_box
from shapely.ops import unary_union
import networkx as nx

#########################
# Core Pathfinding Functions
#########################

def create_free_area(box_coords, obstacles):
    """Create free area by subtracting obstacles from survey box."""
    survey_area = shapely_box(*box_coords)
    obstacle_polys = [shapely_box(*obs) for obs in obstacles]
    if obstacle_polys:
        obstacles_union = unary_union(obstacle_polys)
        return survey_area.difference(obstacles_union)
    return survey_area

def generate_grid_waypoints(box_coords, obstacles, grid_res):
    """Generate grid points with obstacle checking."""
    world_width = box_coords[2] - box_coords[0]
    world_height = box_coords[3] - box_coords[1]
    
    # Generate grid coordinates
    x_coords = np.arange(grid_res/2 + box_coords[0], box_coords[2], grid_res)
    y_coords = np.arange(grid_res/2 + box_coords[1], box_coords[3], grid_res)
    
    grid = []
    visited = []
    free_area = create_free_area(box_coords, obstacles)
    
    for x in x_coords:
        row = []
        row_visited = []
        for y in y_coords:
            pt = Point(x, y)
            if free_area.contains(pt):
                row.append(pt)
                row_visited.append(False)
            else:
                row.append(pt)
                row_visited.append(True)
        grid.append(row)
        visited.append(row_visited)
    
    return np.array(grid), np.array(visited), free_area

def build_global_visibility_graph(free_area, obstacles):
    """Build global visibility graph with obstacle and boundary vertices"""
    nodes = set()
    
    # Add free area boundary vertices
    for coord in free_area.exterior.coords:
        nodes.add(Point(coord))
    
    # Add obstacle vertices
    for obs in obstacles:
        obs_poly = shapely_box(*obs)
        for coord in obs_poly.exterior.coords:
            pt = Point(coord)
            if free_area.covers(pt):
                nodes.add(pt)
    
    nodes = list(nodes)
    G = nx.Graph()
    
    # Add nodes
    for node in nodes:
        G.add_node(node)
    
    # Connect visible nodes
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            a = nodes[i]
            b = nodes[j]
            line = LineString([a, b])
            if free_area.covers(line):
                dist = a.distance(b)
                G.add_edge(a, b, weight=dist)
                G.add_edge(b, a, weight=dist)
    
    return G

def find_path_global(G, free_area, start_pt, end_pts):
    """Find path between two points using global visibility graph"""
    # convert end_pts to Point objects
    end_pts = [Point(pt) for pt in end_pts]

    # copy and add start_pt node and add end_pts
    temp_G = G.copy()
    temp_G.add_node(start_pt)
    for pt in end_pts:
        temp_G.add_node(pt)
    
    # connect start_pt to every existing node it sees
    for node in G.nodes:
        line = LineString([start_pt, node])
        if free_area.covers(line):
            d = Point(start_pt).distance(Point(node))
            temp_G.add_edge(start_pt, node, weight=d)
            temp_G.add_edge(node, start_pt, weight=d)

    # connect end_pts to every existing node they see
    for end_pt in end_pts:
        for node in temp_G.nodes:
            if node.equals(end_pt):
                continue
            line = LineString([end_pt, node])
            if free_area.covers(line):
                d = Point(end_pt).distance(Point(node))
                temp_G.add_edge(end_pt, node, weight=d)
                temp_G.add_edge(node, end_pt, weight=d)


    # run one Dijkstra from start_pt
    lengths, preds = nx.single_source_dijkstra(temp_G, source=start_pt, weight='weight')
    
    # among end_pts, pick the reachable one with minimal length
    best_end = None
    best_len = float('inf')
    for e in end_pts:
        if e in lengths and lengths[e] < best_len:
            best_len = lengths[e]
            best_end = e
    
    if best_end is None:
        return None
    
    return preds[best_end]

def grid_tour(start_pt, grid, visited_mask, free_area, obstacles, global_G):
    """Generate path using greedy direction-prioritized traversal."""

    # Find the closest unvisited point from the start point using the visibility graph
    path = [start_pt]
    unvisited = [grid[i, j] for i, j in zip(*np.where(~visited_mask))]
    shortest_path = find_path_global(global_G, free_area, start_pt, unvisited)
    if shortest_path:
        closest_point = Point(shortest_path[-1])
        start_ij = np.argwhere(grid == closest_point)[0]
        path.extend([Point(coord) for coord in shortest_path[1:]])
        visited_mask[start_ij[0], start_ij[1]] = True
        

    directions = [(-1,0), (1,0), (0,-1), (0,1)]
    height, width = grid.shape[0], grid.shape[1]
    current_i, current_j = start_ij
    current_pt = grid[current_i, current_j]
    visited_mask[current_i, current_j] = True

    def get_unvisited_points():
        return [grid[i,j] for i,j in zip(*np.where(~visited_mask))]
    
    def check_visited(i, j):
        return i >= 0 and i < height and j >= 0 and j < width and not visited_mask[i,j]
    
    def check_visited_orthogonal(i, j, dir):
        dirs = [directions[3 - dir], directions[3 - (dir ^ 1)]]
        for di, dj in dirs:
            ni, nj = i + di, j + dj
            if ni < 0 or ni >= height or nj < 0 or nj >= width or visited_mask[ni, nj]:
                return True
        return False
    
    def scan_direction(di, dj):
        ni, nj = current_i, current_j 
        count = 0
        while True:
            ni += di
            nj += dj
            if not check_visited(ni, nj) or not check_visited_orthogonal(ni, nj, directions.index((di, dj))):
                break
            count += 1
        return count

    while True:
        # Try current direction first
        best_dir = None
        max_count = 0
        current_pt = grid[current_i, current_j]
        
        # Find best direction
        for idx, (di, dj) in enumerate(directions):
            cnt = scan_direction(di, dj)
            if cnt > max_count:
                max_count = cnt
                best_dir = idx
                
        if best_dir is not None and max_count > 0:
            di, dj = directions[best_dir]
            for _ in range(max_count):
                current_i += di
                current_j += dj
                path.append(grid[current_i, current_j])
                assert(not visited_mask[current_i, current_j])
                visited_mask[current_i, current_j] = True
            # Check if we can turn once in an orthogonal direction
            orthogonal_dirs = [directions[3 - best_dir], directions[3 - (best_dir ^ 1)]]
            for ortho_di, ortho_dj in orthogonal_dirs:
                next_i, next_j = current_i + ortho_di, current_j + ortho_dj
                if check_visited(next_i, next_j):
                    current_i, current_j = next_i, next_j
                    path.append(grid[current_i, current_j])
                    assert(not visited_mask[current_i, current_j])
                    visited_mask[current_i, current_j] = True
                    break
        else:
            # Visibility graph pathfinding
            unvisited = get_unvisited_points()
            if not unvisited:
                break

            # Find nearest unvisited point using visibility graph
            shortest_path = find_path_global(global_G, free_area, current_pt, unvisited)
            if shortest_path:
                # Add path to our tour
                path.extend([Point(coord) for coord in shortest_path[1:]])
                
                # Update visited mask and current position
                for coord in shortest_path[1:]:
                    pt = Point(coord)
                    for i in range(height):
                        for j in range(width):
                            if grid[i,j].equals(pt):
                                current_i, current_j = i, j
                                visited_mask[i,j] = True
                                break
            else:
                break  # No path found
        

    # Return to start
    return_path = find_path_global(
        global_G, free_area,
        path[-1],
        [path[0]]
    )
    if return_path:
        path.extend([Point(coord) for coord in return_path[1:]])

    # print(path)

    return LineString([(p.x, p.y) for p in path])

#########################
# Visualization
#########################

def plot_paths(box_coords, obstacles, grid, paths, bases, filename):
    """Visualize paths and save to file."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot survey area
    free_area = create_free_area(box_coords, obstacles)
    x, y = free_area.exterior.xy
    ax.fill(x, y, alpha=0.1, fc='green', ec='darkgreen')
    
    # Plot obstacles
    for obs in obstacles:
        poly = shapely_box(*obs)
        x, y = poly.exterior.xy
        ax.fill(x, y, fc='gray', ec='black', alpha=0.7)
    
    # Plot grid points
    for row in grid:
        for pt in row:
            if free_area.covers(pt):
                ax.plot(pt.x, pt.y, 'o', markersize=2, color='blue', alpha=0.3)
    
    # Plot paths
    colors = ['red', 'blue']
    labels = ['Drone 1', 'Drone 2']
    stats = []
    
    for i, path in enumerate(paths):
        if path:
            x, y = path.xy
            ax.plot(x, y, color=colors[i], linewidth=2, label=labels[i])
            length = path.length
            turns = calculate_total_turn_angle(path)
            stats.append(f"{labels[i]}: {length:.1f} units, {turns:.1f}° turns")
    
    # Plot bases
    for i, base in enumerate(bases):
        ax.plot(base[0], base[1], '*', markersize=12, 
                color=colors[i], markeredgecolor='black', label=f'Base {i+1}')
    
    # Add stats
    if stats:
        ax.text(0.05, 0.95, '\n'.join(stats), transform=ax.transAxes,
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
    
    ax.set_xlim(box_coords[0], box_coords[2])
    ax.set_ylim(box_coords[1], box_coords[3])
    ax.set_aspect('equal')
    ax.set_title("Multi-Drone Survey Coverage Path")
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

def calculate_total_turn_angle(path):
    """Calculate total turning angle in degrees."""
    coords = np.array(path.coords)
    vectors = np.diff(coords, axis=0)
    angles = []
    
    for i in range(1, len(vectors)):
        v1 = vectors[i-1]
        v2 = vectors[i]
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm == 0:
            continue
        cos_theta = np.dot(v1, v2) / norm
        angles.append(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))
        
    return sum(angles)

#########################
# Main Execution
#########################

# Parameters
box_coords = (0, 0, 800, 620)
obstacles = [
    (350, 120, 426, 220),
    (485, 280, 525, 378),
    (350, 20, 400, 70),
    (175, 420, 225, 470),
]
obs_tolerance = 5
survey_radius = 25
grid_res = survey_radius * 2
bases = [(450, 200), (475, 200)]  # Initial base points
split_x = 460  # Initial split line

# Create expanded obstacles
big_obstacles = [
    (x1-obs_tolerance, y1-obs_tolerance, x2+obs_tolerance, y2+obs_tolerance)
    for x1, y1, x2, y2 in obstacles
]

# Generate grid
grid, visited_mask, free_area = generate_grid_waypoints(box_coords, big_obstacles, grid_res)

# Create region masks
grid_x = np.array([[pt.x for pt in row] for row in grid])
left_region_mask = (grid_x > split_x) | visited_mask
right_region_mask = (grid_x <= split_x) | visited_mask
global_G = build_global_visibility_graph(free_area, big_obstacles)

# Generate paths
left_path = grid_tour(Point(bases[0]), grid, np.copy(left_region_mask), free_area, big_obstacles, global_G)
right_path = grid_tour(Point(bases[1]), grid, np.copy(right_region_mask), free_area, big_obstacles, global_G)

# Visualize
plot_paths(box_coords, obstacles, grid, [left_path, right_path], bases, f"two_drones_survey_path_grid_{survey_radius}.png")