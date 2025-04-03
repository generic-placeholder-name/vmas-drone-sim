import torch
from vmas.simulator.core import Landmark
import numpy as np

class Graph():
    """A Graph represents a collection of waypoints connected by edges."""
    def __init__(self, waypoints=None, edges=None):
        """
        Args:
            waypoints (list): A list of Waypoints to initialize the graph with.
            edges (list): A list of Edges to initialize the graph with.
        """
        self._waypoints = waypoints if waypoints is not None else []
        self._edges = edges if edges is not None else []

        assert all(isinstance(waypoint, Waypoint) for waypoint in self._waypoints), "All waypoints must be instances of Waypoint"
        assert all(isinstance(edge, Edge) for edge in self._edges), "All edges must be instances of Edge"

    @property
    def waypoints(self):
        return self._waypoints

    @property
    def edges(self):
        return self._edges

    def generate_edges(self):
        """Generate all possible edges between waypoints."""
        for i, node1 in enumerate(self._waypoints):
            for j, node2 in enumerate(self._waypoints):
                if i != j and self.get_edge(node1, node2) is None:
                    assert node1 != node2, "repeated waypoints in self._waypoints"
                    edge = Edge(node1, node2)
                    self.add_edge(edge)
    
    def remove_edges(self, edges : list):
        """Remove unwanted edges from graph list"""
        # Had to put import here, otherwise there was a circular dependency
        from test import envConfig  # Get environment config to get penalty areas
        print(f"Initial edges: {len(edges)}")
        penalties = envConfig["penaltyAreas"]
        self._edges = [edge for edge in edges if self.edge_valid(edge, penalties)]
        print(f"Edges after removal: {len(self._edges)}")

    def edge_valid(self, edge, penalties, margin=0.08):
        """Ensure edge is valid, i.e. does not go through penalty area or too close to penalty area"""
        w1 = edge.node1
        w2 = edge.node2
        w1_valid = self.waypoint_valid(w1, penalties)
        w2_valid = self.waypoint_valid(w2, penalties)
        if w1_valid and w2_valid:
            for penalty in penalties:
                # New top left and bottom right with padding
                tl = torch.tensor([penalty["topLeft"][0],penalty["topLeft"][1]]) - torch.tensor([margin, margin])
                br = torch.tensor([penalty["bottomRight"][0],penalty["bottomRight"][1]]) + torch.tensor([margin, margin])

                # Rectangle corners (clockwise order)
                corners = [tl, torch.tensor([br[0], tl[1]]), br, torch.tensor([tl[0], br[1]])]

                # Check intersection between the line and each of the rectangle's sides
                for i in range(4):
                    if self.do_lines_intersect(w1.point, w2.point, corners[i], corners[(i + 1) % 4]):
                        print(f"Edge {w1} to {w2} intersects with penalty area")
                        return False
            # No intersection, return True for valid
            return True
        else:
            if not w1_valid:
                print(f"Waypoint {w1} invalid")
            if not w2_valid:
                print(f"Waypoint {w2} invalid")
            return False
    
    def do_lines_intersect(self, p1, p2, q1, q2):
        # Calculate cross products to see if segments intersect
        d1 = np.cross(np.array(p2) - np.array(p1), np.array(q1) - np.array(p1))
        d2 = np.cross(np.array(p2) - np.array(p1), np.array(q2) - np.array(p1))
        d3 = np.cross(np.array(q2) - np.array(q1), np.array(p1) - np.array(q1))
        d4 = np.cross(np.array(q2) - np.array(q1), np.array(p2) - np.array(q1))

        return (d1 * d2 < 0) and (d3 * d4 < 0)

    def waypoint_valid(self, waypoint, penalties, margin=0.08):
        """Ensure waypoint is valid, i.e. not in penalty area or too close to penalty area"""
        point = waypoint.point
        for penalty in penalties:
            top_left_x = penalty["topLeft"][0] - margin
            bottom_right_x = penalty["bottomRight"][0] + margin
            top_left_y = penalty["topLeft"][1] - margin
            bottom_right_y = penalty["bottomRight"][1] + margin
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

    def get_edge(self, node1, node2):
        """Get an edge by its two connected waypoints."""
        temp_edge = Edge(node1, node2)
        for edge in self._edges:
            if edge == temp_edge:
                return edge
        return None

    def get_edges(self, node):
        """Get all edges connected to a waypoint."""
        return [edge for edge in self._edges if edge.node1 == node or edge.node2 == node]

    def get_neighbors(self, node, exclude_traversed=False):
        """Get all neighbors of a waypoint."""
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
    
    def __str__(self) -> str:
        str = "Graph:\n"
        for edge in self._edges:
            str += f"{edge}\n"
        return str
    

class Waypoint():
    """A Waypoint represents a point in 2D space with an associated vmas landmark."""
    def __init__(self, point: torch.Tensor, landmark: Landmark, reward_radius=0.01, dtype=torch.float32):
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
        return f"{self.landmark.name}({self._point})"


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
        self._length = torch.tensor(length, dtype=torch.float32) if length is not None else self.estimate_length()
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
        assert value > 0, "Weight must be positive"
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

    def estimate_length(self):
        """Estimate the length of the edge based on the distance between the two waypoints."""
        return torch.linalg.vector_norm(self._node2.point - self._node1.point)
    
    def __str__(self):
        return f"Edge({self._node1}, {self._node2}) (weight: {self.weight})"
    
    def __eq__(self, other):
        return (self.node1 == other.node1 and self.node2 == other.node2) or (self.node1 == other.node2 and self.node2 == other.node1)

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

        assert isinstance(edge, Edge), "edge must be an instance of Edge"
        assert isinstance(previous_edge, Edge), "previous_edge must be an instance of Edge"
        assert get_node_in_common(previous_edge, edge) is not None, "The edges must be connected"
        assert previous_edge != edge, "The edges must not be the same"
        assert self._weight > 0, "Weight must be positive"

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
        assert value > 0, "Weight must be positive"
        self._weight = torch.tensor(value, dtype=torch.float32)

    def add_weight(self, value):
        self._weight += torch.tensor(value, dtype=torch.float32)
        if self._weight < 0:
            self._weight = torch.tensor(0.0, dtype=torch.float32)

    def angle(self):
        """Calculate the angle (in radians) between the two edges."""
        # Get the direction vectors of the edges
        dir1 = self._edge._node2.point - self._edge._node1.point
        dir2 = self._previous_edge._node2.point - self._previous_edge._node1.point
        
        # Normalize the direction vectors
        dir1 = dir1 / torch.linalg.vector_norm(dir1)
        dir2 = dir2 / torch.linalg.vector_norm(dir2)
        
        # Calculate the angle using the dot product
        cos_theta = torch.dot(dir1, dir2)
        return torch.acos(torch.clamp(cos_theta, -1.0, 1.0))

def get_node_in_common(edge1, edge2):
    """Get the node in common between two edges."""
    if edge1.node1 == edge2.node1 or edge1.node1 == edge2.node2:
        return edge1.node1
    elif edge1.node2 == edge2.node1 or edge1.node2 == edge2.node2:
        return edge1.node2
    else:
        return None