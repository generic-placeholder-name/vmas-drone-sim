import torch
from vmas.simulator.core import Landmark

class Waypoint():
    """A Waypoint represents a point in 2D space with an associated vmas landmark."""
    def __init__(self, point: torch.Tensor, landmark: Landmark, reward_radius=0.01, dtype=torch.float32):
        self._point = point
        self._landmark = landmark
        self._reward_radius = reward_radius
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
    def _length(self):
        return self._length
    
    @_length.setter
    def _length(self, value):
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

class Elbow():
    """An Elbow is a connection between two edges, where the first edge ends at the second edge's start node."""
    def __init__(self, edge: Edge, previous_edge: Edge, weight=1.0):
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
        assert edge._node1 == previous_edge._node2, "The edges must be connected"
        assert edge._node2 != previous_edge._node1, "The edges must not be the same"
        assert edge._node2.landmark != previous_edge._node1.landmark, "The edges must have different landmarks"
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
