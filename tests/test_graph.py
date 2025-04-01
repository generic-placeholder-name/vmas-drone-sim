import pytest
from graph import *
from vmas.simulator.core import Landmark
from vmas.simulator.core import Sphere
from vmas.simulator.core import Color
from torch import Tensor

#region Waypoint Tests
@pytest.fixture
def landmark1():
    return Landmark("A", shape=Sphere(radius=0.1))

@pytest.fixture
def landmark2():
    return Landmark("B", shape=Sphere(radius=0.1))

@pytest.fixture
def landmark3():
    return Landmark("C", shape=Sphere(radius=0.1))

@pytest.fixture
def landmark4():
    return Landmark("D", shape=Sphere(radius=0.1))

@pytest.fixture
def waypoint1(landmark1):
    return Waypoint(Tensor([1.0, 1.0]), landmark1)

@pytest.fixture
def waypoint2(landmark2):
    return Waypoint(Tensor([2.0, 3.0]), landmark2)

@pytest.fixture
def waypoint3(landmark3):
    return Waypoint(Tensor([3.0, 2.5]), landmark3)

@pytest.fixture
def waypoint4(landmark4):
    return Waypoint(Tensor([6.0, 1.0]), landmark4)

# Visual representation of the waypoints:
# 
#
#   A                   D
#
#
#           C  
#       B

def test_get_waypoint_point(waypoint1):
    assert torch.equal(waypoint1.point, torch.tensor([1.0, 1.0]))

def test_get_waypoint_landmark(waypoint1, landmark1):
    assert waypoint1.landmark == landmark1

def test_get_waypoint_reward_radius(waypoint1):
    assert waypoint1.reward_radius == 0.01

def test_get_waypoint_traversed(waypoint1):
    assert not waypoint1.traversed

#endregion

#region Edge tests
@pytest.fixture
def edge_w1_w2(waypoint1, waypoint2):
    return Edge(waypoint1, waypoint2)

@pytest.fixture
def edge_w1_w3(waypoint1, waypoint3):
    return Edge(waypoint1, waypoint3)

@pytest.fixture
def edge_w1_w4(waypoint1, waypoint4):
    return Edge(waypoint1, waypoint4)

def test_get_edge_node1(edge_w1_w2, waypoint1):
    assert edge_w1_w2.node1 == waypoint1

def test_get_edge_node2(edge_w1_w2, waypoint2):
    assert edge_w1_w2.node2 == waypoint2

def test_edge_weight(edge_w1_w2):
    assert edge_w1_w2.weight == 1.0
    edge_w1_w2.add_weight(2.0)
    assert edge_w1_w2.weight == 3.0
    edge_w1_w2.add_weight(-5)
    assert edge_w1_w2.weight == 0.0

def test_edge_length(edge_w1_w2, edge_w1_w3, edge_w1_w4):
    assert edge_w1_w2.length == 2.23606797749979
    assert edge_w1_w3.length == 2.5
    assert edge_w1_w4.length == 5.0

#endregion

#region Elbow tests
@pytest.fixture
def edge_w2_w3(waypoint2, waypoint3):
    return Edge(waypoint2, waypoint3)

@pytest.fixture
def elbow1(edge_w1_w2, edge_w2_w3):
    return Elbow(edge_w1_w2, edge_w2_w3)

def test_elbow_angle(elbow1, edge_w1_w2, edge_w2_w3, edge_w1_w3):
    assert elbow1.angle() == torch.arccos((edge_w1_w2.length**2 + edge_w2_w3.length**2 - edge_w1_w3.length**2) / (2 * edge_w1_w2.length * edge_w2_w3.length))

#endregion

#region Graph tests

@pytest.fixture
def graph_with_waypoints(waypoint1, waypoint2, waypoint3, waypoint4):
    return Graph([waypoint1, waypoint2, waypoint3, waypoint4])

@pytest.fixture
def graph_with_generated_edges(graph_with_waypoints):
    graph_with_waypoints.generate_edges()
    return graph_with_waypoints


def test_graph_generate_edges(graph_with_generated_edges):
    assert len(graph_with_generated_edges.edges) == 6
    assert str(graph_with_generated_edges) == ("Graph:" +
                                               "\nEdge(A(tensor([1., 1.])), B(tensor([2., 3.])))" +
                                               "\nEdge(A(tensor([1., 1.])), C(tensor([3.0000, 2.5000])))" +
                                               "\nEdge(A(tensor([1., 1.])), D(tensor([6., 1.])))" +
                                               "\nEdge(B(tensor([2., 3.])), C(tensor([3.0000, 2.5000])))" +
                                               "\nEdge(B(tensor([2., 3.])), D(tensor([6., 1.])))" +
                                               "\nEdge(C(tensor([3.0000, 2.5000])), D(tensor([6., 1.])))\n")
    
def test_graph_get_neighbors_untraversed(graph_with_generated_edges, waypoint1, waypoint2, waypoint3, waypoint4):
    node_neighbors, edge_neighbors = graph_with_generated_edges.get_neighbors(waypoint3)
    assert len(node_neighbors) == 3
    assert len(edge_neighbors) == 3
    assert waypoint1 in node_neighbors
    assert waypoint2 in node_neighbors
    assert waypoint4 in node_neighbors
    assert node_neighbors[0] in edge_neighbors[0].nodes and waypoint3 in edge_neighbors[0].nodes
    assert node_neighbors[1] in edge_neighbors[1].nodes and waypoint3 in edge_neighbors[1].nodes
    assert node_neighbors[2] in edge_neighbors[2].nodes and waypoint3 in edge_neighbors[2].nodes

def test_graph_get_neighbors_traversed(graph_with_generated_edges, waypoint1, waypoint2, waypoint3, waypoint4):
    waypoint1.traversed = True
    node_neighbors, edge_neighbors = graph_with_generated_edges.get_neighbors(waypoint3)
    assert len(node_neighbors) == 3
    assert len(edge_neighbors) == 3

    node_neighbors, edge_neighbors = graph_with_generated_edges.get_neighbors(waypoint3, exclude_traversed=True)
    assert waypoint2 in node_neighbors
    assert waypoint4 in node_neighbors
    assert node_neighbors[0] in edge_neighbors[0].nodes and waypoint3 in edge_neighbors[0].nodes
    assert node_neighbors[1] in edge_neighbors[1].nodes and waypoint3 in edge_neighbors[1].nodes

    waypoint2.traversed = True
    node_neighbors, edge_neighbors = graph_with_generated_edges.get_neighbors(waypoint3, exclude_traversed=True)
    assert len(node_neighbors) == 1
    assert len(edge_neighbors) == 1
    assert waypoint4 in node_neighbors
    assert node_neighbors[0] in edge_neighbors[0].nodes and waypoint3 in edge_neighbors[0].nodes

    waypoint4.traversed = True
    node_neighbors, edge_neighbors = graph_with_generated_edges.get_neighbors(waypoint3, exclude_traversed=True)
    assert len(node_neighbors) == 0
    assert len(edge_neighbors) == 0

def test_graph_reset_traversed(graph_with_generated_edges, waypoint1, waypoint2, waypoint3, waypoint4):
    waypoint1.traversed = True
    waypoint2.traversed = True
    waypoint3.traversed = True
    waypoint4.traversed = True
    graph_with_generated_edges.reset_traversed()
    assert not waypoint1.traversed
    assert not waypoint2.traversed
    assert not waypoint3.traversed
    assert not waypoint4.traversed
    
#endregion