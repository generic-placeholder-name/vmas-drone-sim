import torch
from util.priority_queue import PriorityQueue
from vmas.simulator.core import Landmark
import numpy as np
import copy

class Waypoint():
    """A Waypoint represents a point in 2D space with an associated vmas landmark."""
    def __init__(self, point: torch.Tensor, landmark: Landmark | None=None, reward_radius=0.01, dtype=torch.float32):
        self._point = point
        self._landmark = landmark
        self._reward_radius = reward_radius
        self._traversed = False
        assert self._point.shape == (2,), "Point must be a 2D tensor"
        assert self._point.dtype == dtype, f"Point must be a {dtype} tensor"
        assert isinstance(self._landmark, Landmark), "landmark must be an instance of Landmark"

    @property
    def point(self):
        return self._point
    
    @property
    def landmark(self):
        return self._landmark
    
    @property
    def reward_radius(self):
        return self._reward_radius
    
    @property
    def traversed(self):
        return self._traversed
    
    @traversed.setter
    def traversed(self, value):
        assert isinstance(value, bool), "traversed must be a boolean"
        self._traversed = value
    
    def __str__(self) -> str:
        return f"{self.landmark.name if self.landmark is not None else "None"}({self._point})"


class Edge():
    """
    An Edge represents a connection between two Waypoints with a specified length.
    """
    def __init__(self, node1: Waypoint, node2: Waypoint, length=None, weight=1.0):
        """
        Args:
            node1 (Waypoint): The starting waypoint of the edge.
            node2 (Waypoint): The ending waypoint of the edge.
            length (float): The length of the edge (distance between the two waypoints).
            weight (float): The weight (or favorability) of the edge.
        """
        self._node1 = node1
        self._node2 = node2
        self._length = torch.tensor(length, dtype=torch.float32) if length is not None else self.calculate_length()
        self._weight = torch.tensor(weight, dtype=torch.float32)

        assert self._length > 0, "Length must be positive"
        assert self._weight > 0, "Weight must be positive"
        assert isinstance(node1, Waypoint), "node1 must be an instance of Waypoint"
        assert isinstance(node2, Waypoint), "node2 must be an instance of Waypoint"
        assert node1 != node2, "node1 and node2 must be different Waypoints"
        assert node1.landmark != node2.landmark, "node1 and node2 must have different landmarks"

    @property
    def node1(self):
        return self._node1
    
    @property
    def node2(self):
        return self._node2
    
    @property
    def nodes(self):
        return self._node1, self._node2
    
    @property
    def weight(self):
        return self._weight
    
    @weight.setter
    def weight(self, value):
        assert value > 0, f"Weight must be positive. Got {value}."
        self._weight = torch.tensor(value, dtype=torch.float32)

    def add_weight(self, value):
        self._weight += torch.tensor(value, dtype=torch.float32)
        if self._weight < 0:
            self._weight = torch.tensor(0.0, dtype=torch.float32)

    @property
    def length(self):
        return self._length
    
    @length.setter
    def length(self, value):
        """
        Update the length of the edge, just in case it is different from the estimated length (shortest path).
        This can happen if there is an obstacle between the two waypoints.
        Args:
            value (float): The new length of the edge.
        """
        assert value > 0, "Length must be positive"
        self._length = torch.tensor(value, dtype=torch.float32)

    def calculate_length(self):
        """Estimate the length of the edge based on the distance between the two waypoints."""
        return torch.linalg.vector_norm(self._node2._point - self._node1._point)
    
    def __str__(self):
        return f"Edge({self._node1}, {self._node2}) (weight: {self.weight})"
    
    def __eq__(self, other):
        return (self.node1 == other.node1 and self.node2 == other.node2) or (self.node1 == other.node2 and self.node2 == other.node1)


class Graph():
    """A Graph represents a collection of waypoints connected by edges."""
    def __init__(self, waypoints=None, edges=None, margin=0.):
        """
        Args:
            waypoints (list): A list of Waypoints to initialize the graph with.
            edges (list): A list of Edges to initialize the graph with.
        """
        self._waypoints = waypoints if waypoints is not None else []
        self._edges = edges if edges is not None else []
        self._margin = margin
        self._bad_edges_to_obstacles = {} # e.g.: {edge1: [obstacle1, obstacle3], edge2: [obstacle3]}

        assert all(isinstance(waypoint, Waypoint) for waypoint in self._waypoints), "All waypoints must be instances of Waypoint"
        assert all(isinstance(edge, Edge) for edge in self._edges), "All edges must be instances of Edge"

    @property
    def waypoints(self):
        return self._waypoints

    @property
    def edges(self):
        return self._edges
    
    @property
    def margin(self):
        return self._margin
    
    @margin.setter
    def margin(self, value):
        """Set the margin for obstacle avoidance."""
        assert value >= 0, "Margin must be non-negative"
        self._margin = value

    def extend_graph(self, waypoints: list[Waypoint], edges: list[Edge]):
        """
        Extend the graph by adding new waypoints and edges.

        :param waypoints: A list of waypoints.
        :param edges: A list of edges.
        """
        assert all(isinstance(w, Waypoint) for w in waypoints)
        assert all(isinstance(e, Edge) for e in edges)
        assert all(w not in self._waypoints for w in waypoints)
        assert all(e not in self._edges for e in edges)
        assert all(e.node1 in self._waypoints + waypoints and e.node2 in self._waypoints + waypoints for e in edges), "Edges must connect waypoints that exist in the graph."
        self._waypoints.extend(waypoints)
        self._edges.extend(edges)

    def generate_edges(self, penalty_areas, generate_alternative_routes): # TODO: Have it cal self.edge_valid so that it doesn't add an edge that goes through obstacle and adds algernative routes
        """Generate all possible edges between waypoints."""
        for i, node1 in enumerate(self._waypoints):
            for j, node2 in enumerate(self._waypoints):
                if i != j and self.get_edge(node1, node2) is None:
                    assert node1 != node2, "repeated waypoints in self._waypoints"
                    edge = Edge(node1, node2)
                    if self.edge_valid(edge, penalty_areas, create_path_around_penalties=generate_alternative_routes):
                        self.add_edge(edge)
    
    def remove_edges(self, bad_edges: list):
        """Remove unwanted edges from graph list"""
        print(f"Initial edges: {len(self._edges)}")
        good_edges = [edge for edge in self._edges if edge not in bad_edges]
        self._edges = good_edges
        print(f"Edges after removal: {len(self._edges)}")

    def edge_valid(self, edge: Edge, penalties: list[dict], create_path_around_penalties: bool):
        """
        Ensure edge is valid, i.e. does not go through penalty area or too close to penalty area.
        
        :param edge: Edge to be validated.
        :param penalties: List of dictionaries which define where penalty areas are.
        :param create_path_around_penalties: Boolean indicating whether or not to create an alternative path
        from waypoint 1 to waypoint 2 if a direct path would cause a collision with at least one penalty area.
        This is accomplished by adding new waypoints and edges that go around penalty areas, optimized with A*.
        :return: True if `edge` does not cause a collision or get too close to penalty areas, False otherwise.
        """
        direct_path_available = True
        w1 = edge.node1
        w2 = edge.node2
        w1_valid = self.waypoint_valid(w1, penalties)
        w2_valid = self.waypoint_valid(w2, penalties)
        alternate_path_waypoints = []
        if w1_valid and w2_valid:
            for penalty in penalties:
                # New top left and bottom right with padding
                buffer_vector = torch.tensor([self.margin / torch.tensor(2).sqrt(), self.margin / torch.sqrt(torch.tensor(2))])
                assert buffer_vector.pow(2).sum().sqrt() == self.margin, f"Expected the magnitude of `buffer_vector` to be equal to the margin: {self.margin}. Got: {buffer_vector.pow(2).sum().sqrt()}."
                tl = torch.tensor([penalty["topLeft"][0],penalty["topLeft"][1]]) - buffer_vector
                br = torch.tensor([penalty["bottomRight"][0],penalty["bottomRight"][1]]) + buffer_vector

                # Rectangle vertices (clockwise order)
                vertices = [tl, torch.tensor([br[0], tl[1]]), br, torch.tensor([tl[0], br[1]])]

                # Check intersection between the line and each of the rectangle's sides
                for i in range(4):
                    if self.lines_intersect(w1.point, w2.point, vertices[i], vertices[(i + 1) % 4]):
                        print(f"Edge {w1} to {w2} intersects with penalty area")
                        direct_path_available = False
                        potential_waypoints = [Waypoint(vertices[i], None) for i in vertices]
                        for w in potential_waypoints:
                            if self.waypoint_valid(w, penalties):
                                alternate_path_waypoints.append(w)
            # No intersection, return True for valid
            if direct_path_available:
                return True
            else:
                #self._bad_edges_to_obstacles[edge] = extended_obstacles_in_way # TODO: This instance variable may be unecessary, consider removing it
                if create_path_around_penalties:
                    potential_graph = copy.deepcopy(self)
                    potential_graph.extend_graph(alternate_path_waypoints, edges=[])
                    potential_graph.generate_edges(penalties, generate_alternative_routes=False) # Excludes edges that could cause a collision
                    a_star_path = self.find_optimal_path_AStar(w1, w2, potential_graph)
                    if a_star_path is not None:
                        waypoints, edges = a_star_path[0], a_star_path[1]
                        self.extend_graph(waypoints, edges) # TODO: Currently, will throw error if try to pass in waypoints and edges that already existed. Modify extend_graph.
                    else:
                        print(f"Unable to generate an alternative path from waypoint {w1} to {w2}.")
                return False
        else:
            if not w1_valid:
                print(f"Waypoint {w1} invalid")
            if not w2_valid:
                print(f"Waypoint {w2} invalid")
            return False
        
    def find_optimal_path_AStar(self, start: Waypoint, goal: Waypoint, graph: "Graph") -> tuple[list[Waypoint], list[Edge]] | None:
        """
        Use A* algorithm to find a path around obstacles through intermediate waypoints.
        If a path is found, return True and update the edge with that path.
        If no path is found, return False.
        """
        assert isinstance(start, Waypoint)
        assert isinstance(goal, Waypoint)
        frontier = PriorityQueue()
        frontier.put(start, 0.)
        came_from = dict()
        cost_so_far = dict()
        came_from[start] = None
        cost_so_far[start] = 0.
        current = None

        while not frontier.empty():
            current = frontier.get()
            assert isinstance(current, Waypoint)
            if current == goal:
                break

            for next, _ in graph.get_neighbors(current):
                assert isinstance(next, Waypoint)
                edge = graph.get_edge(current, next)
                assert isinstance(edge, Edge)
                edge_cost = edge.length
                new_cost = cost_so_far[current] + edge_cost
                
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + graph.distance(next, goal) # f(n) = g(n) + h(n)
                    frontier.put(next, priority)
                    came_from[next] = current
        

        if current != goal:
            return None
        assert isinstance(current, Waypoint)
        waypoints = [current]
        edges = []
        while current != start:
            previous = came_from[current]
            edge = graph.get_edge(previous, current)
            assert isinstance(edge, Edge)
            edges.append(edge)
            current = previous
            waypoints.append(current)
        
        assert all(isinstance(w, Waypoint) for w in waypoints)
        assert all(isinstance(e, Edge) for e in edges)
        
        return waypoints, edges

    def lines_intersect(self, p1, p2, q1, q2):
        # Calculate cross products to see if segments intersect
        def to_3d(vec):
            if hasattr(vec, 'numpy'):
                vec = vec.numpy()
            elif not isinstance(vec, np.ndarray):
                vec = np.asarray(vec)
            return np.array([vec[0], vec[1], 0.0])

        v1 = to_3d(p2 - p1)
        v2 = to_3d(q1 - p1)
        d1 = np.cross(v1, v2)[2]

        v1 = to_3d(p2 - p1)
        v2 = to_3d(q2 - p1)
        d2 = np.cross(v1, v2)[2]

        v1 = to_3d(q2 - q1)
        v2 = to_3d(p1 - q1)
        d3 = np.cross(v1, v2)[2]

        v1 = to_3d(q2 - q1)
        v2 = to_3d(p2 - q1)
        d4 = np.cross(v1, v2)[2]

        return (d1 * d2 < 0) and (d3 * d4 < 0)

    def distance(self, node1: Waypoint, node2: Waypoint) -> float:
        """
        Straight-line distance between two waypoints.

        :param node1: A Waypoint different from `node2`.
        :param node2: A Waypoint different from `node1`.
        :return: A float representing the straight-line distance between `node1` and `node2`.
        """
        point1, point2 = node1.point, node2.point
        return torch.linalg.vector_norm(point2 - point1)

    def waypoint_valid(self, waypoint, penalties):
        """Ensure waypoint is valid, i.e. not in penalty area or too close to penalty area"""
        point = waypoint.point
        for penalty in penalties:
            top_left_x = penalty["topLeft"][0] - self.margin
            bottom_right_x = penalty["bottomRight"][0] + self.margin
            top_left_y = penalty["topLeft"][1] - self.margin
            bottom_right_y = penalty["bottomRight"][1] + self.margin
            # Check if waypoint is within penalty area + margin/padding
            if (top_left_x <= point[0].item() <= bottom_right_x) and (top_left_y <= point[1].item() <= bottom_right_y):
                # It is in penalty area, is invalid so return False
                return False
        return True

    def add_waypoint(self, waypoint):
        assert isinstance(waypoint, Waypoint), "waypoint must be an instance of Waypoint"
        self._waypoints.append(waypoint)

    def add_edge(self, edge):
        assert isinstance(edge, Edge), "edge must be an instance of Edge"
        if self.get_edge(edge.node1, edge.node2) is None:
            self._edges.append(edge)

    def get_edge(self, node1: Waypoint, node2: Waypoint) -> Edge | None:
        """
        Get an edge by its two connected waypoints.
        
        :param node1: Waypoint connected to `node2` via an edge.
        :param node2: Waypoint connected to `node1` via an edge.
        :return: An edge connecting `node1` and `node2`.
        """
        temp_edge = Edge(node1, node2)
        for edge in self._edges:
            if edge == temp_edge:
                return edge
        return None

    def get_edges(self, node):
        """Get all edges connected to a waypoint."""
        return [edge for edge in self._edges if edge.node1 == node or edge.node2 == node]

    def get_neighbors(self, node: Waypoint, exclude_traversed: bool=False) -> tuple[list[Waypoint], list[Edge]]:
        """
        Get all neighbors of a waypoint.
        
        :param node: A Waypoint in the graph.
        :return: A tuple where the first element is a list of Waypoint instances and the second element is a list of Edge instances.
        """
        node_neighbors = []
        edge_neighbors = []
        for edge in self.get_edges(node):
            if edge.node1 == node:
                if not (exclude_traversed and edge.node2.traversed):
                    node_neighbors.append(edge.node2)
                    edge_neighbors.append(edge)
            elif edge.node2 == node:
                if not (exclude_traversed and edge.node1.traversed):
                    node_neighbors.append(edge.node1)
                    edge_neighbors.append(edge)
        return node_neighbors, edge_neighbors
    
    def fully_traversed(self):
        """Check if all waypoints in the graph have been traversed."""
        return all(waypoint.traversed for waypoint in self._waypoints)
    
    def reset_traversed(self):
        """Set all waypoints in the graph as not traversed."""
        for waypoint in self._waypoints:
            waypoint.traversed = False

    def rest_weights(self, value=1.0):
        """Reset all edge weights to a given value."""
        for edge in self._edges:
            edge.weight = value

    def reset(self, weight_value=1.0):
        """Reset all waypoints and edges in the graph."""
        self.reset_traversed()
        self.rest_weights(weight_value)

    def get_path_costs(self, edges=None, waypoints=None):
        """Returns total distance and total rotations of a path"""
        if edges is not None and waypoints is not None:
            raise ValueError("Both edges and waypoints were provided when only one of them should be.")
        elif edges is not None:
            return self.get_cost_from_edges_tour(edges)
        elif waypoints is not None:
            return self.get_cost_from_edges_tour(self.get_edges_from_waypoint_tour(waypoints))
        else:
            raise ValueError("edges or (exclusive) waypoints must be provided.")
        
    def get_edges_from_waypoint_tour(self, waypoints):
        """Returns the equivalent waypoint path expressed in edges"""
        edges = []
        previous_waypoint_index = 0
        for i in range(1, len(waypoints)):
            edge = self.get_edge(waypoints[previous_waypoint_index], waypoints[i])
            assert edge is not None, f"Invalid path. No edge connects waypoints {waypoints[previous_waypoint_index]} and {waypoints[i]}"
            edges.append(edge)
            previous_waypoint_index = i
        return edges

    def get_cost_from_edges_tour(self, edges):
        """Returns the total costs from traversing edges path (distance, rotation)"""
        previous_edge = None
        distance = 0
        rotation = 0
        for edge in edges:
            distance += edge.length
            rotation += self.get_rotation(previous_edge, edge)
            previous_edge = edge
        return distance, rotation

    def get_rotation(self, previous_edge, edge):
        """Return degrees rotated after traversing previous edge and current edge"""
        assert previous_edge is None or isinstance(previous_edge, Edge), "previous_edge must be an edge"
        assert isinstance(edge, Edge), "edge must be an edge"
        if previous_edge is None:
            return 0
        else:
            assert previous_edge != edge, f"Can't get rotation between the same edges, {previous_edge} and {edge}"
            return Elbow(previous_edge, edge).rotation()

    def __str__(self) -> str:
        str = "Graph:\n"
        for edge in self._edges:
            str += f"{edge}\n"
        return str
    
class Elbow():
    """An Elbow is a connection between two edges, where the first edge ends at the second edge's start node."""
    def __init__(self, previous_edge: Edge, edge: Edge, weight=1.0):
        """
        Args:
            edge (Edge): The current edge between the current waypoint and target waypoint being considered.
            previous_edge (Edge): The previous edge that was traversed to get to the current waypoint.
            weight (float): The weight (or favorability) of the edge beign considered, taking into account the previous edge.
        """
        self._edge = edge
        self._previous_edge = previous_edge
        self._weight = torch.tensor(weight, dtype=torch.float32)
        self._point1 = None
        self._point2 = None
        self._point3 = None

        assert isinstance(edge, Edge), "edge must be an instance of Edge"
        assert isinstance(previous_edge, Edge), "previous_edge must be an instance of Edge"
        assert get_node_in_common(previous_edge, edge) is not None, "The edges must be connected"
        assert previous_edge != edge, "The edges must not be the same"
        assert self._weight > 0, "Weight must be positive"
        self.set_ordered_points()

    @property
    def edge(self):
        return self._edge
    
    @property
    def previous_edge(self):
        return self._previous_edge
    
    @property
    def weight(self):
        return self._weight
    
    @weight.setter
    def weight(self, value):
        assert value > 0, f"Weight must be positive. Got {value}."
        self._weight = torch.tensor(value, dtype=torch.float32)

    def add_weight(self, value):
        self._weight += torch.tensor(value, dtype=torch.float32)
        if self._weight < 0:
            self._weight = torch.tensor(0.0, dtype=torch.float32)

    def set_ordered_points(self):
        common_node = self.get_common_node()
        self.point1 = self.previous_edge.node1.point if self.previous_edge.node2 == common_node else self.previous_edge.node2.point
        self.point2 = common_node.point
        self.point3 = self.edge.node2.point if self.edge.node1 == common_node else self.edge.node1.point

    def get_common_node(self):
        if self.previous_edge.node1 in self.edge.nodes:
            return self.previous_edge.node1
        elif self.previous_edge.node2 in self.edge.nodes:
            return self.previous_edge.node2
        else:
            raise ValueError(f"No common node found in edges. Nodes found: {self.previous_edge.node1}, {self.previous_edge.node2}, {self.edge.node1}, {self.edge.node2}")
        
    def angle(self):
        """Calculate the angle (in degrees) between the two edges."""
        # Get the direction vectors of the edges
        dir1 = self.point3 - self.point2
        dir2 = self.point1 - self.point2
        
        # Normalize the direction vectors
        dir1 = dir1 / torch.linalg.vector_norm(dir1)
        dir2 = dir2 / torch.linalg.vector_norm(dir2)
        
        # Calculate the angle using the dot product
        cos_theta = torch.dot(dir1, dir2)
        return torch.rad2deg(torch.acos(torch.clamp(cos_theta, -1.0, 1.0)))
    
    def rotation(self):
        """Calculates the exterior angle betwee the two edges in degrees, or the degrees a drone would have to rotate."""
        return 180 - self.angle()

def get_node_in_common(edge1, edge2):
    """Get the node in common between two edges."""
    if edge1.node1 == edge2.node1 or edge1.node1 == edge2.node2:
        return edge1.node1
    elif edge1.node2 == edge2.node1 or edge1.node2 == edge2.node2:
        return edge1.node2
    else:
        return None