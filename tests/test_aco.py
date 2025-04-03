import pytest
from aco import *
import graph
from test_graph import *

def test_heuristic():
    assert heuristic(3, 1) == 1/3
    assert heuristic(3, 2) == (1/3) * (1/2)
    assert heuristic(4, 0) == (1/4) * (1/0.1)
    
@pytest.fixture
def ant1(graph_with_generated_edges, waypoint1):
    return Ant(graph_with_generated_edges, waypoint1)

@pytest.fixture
def ant2(graph_with_generated_edges, waypoint1):
    return Ant(graph_with_generated_edges, waypoint1)

@pytest.fixture
def ant3(graph_with_generated_edges, waypoint1):
    return Ant(graph_with_generated_edges, waypoint1)

def test_ant_get_rotation(ant1, edge_w1_w2, edge_w2_w3):
    assert ant1.get_rotation(edge_w1_w2) == 0
    assert ant1.current_node == edge_w1_w2.node1
    ant1.move_to(edge_w1_w2.node2, edge_w1_w2)
    #ant1.move_to(edge_w2_w3.node2, edge_w2_w3)
    assert ant1.get_rotation(edge_w2_w3) == abs(Elbow(edge_w1_w2, edge_w2_w3).angle())

@pytest.fixture
def edge_w3_w2(waypoint3, waypoint2):
    return Edge(waypoint3, waypoint2)

@pytest.fixture
def edge_w3_w4(waypoint3, waypoint4):
    return Edge(waypoint3, waypoint4)

def test_ant_move_to(ant1, waypoint1, waypoint2, waypoint3, waypoint4, edge_w1_w2, edge_w3_w2, edge_w3_w4):
    assert ant1.current_node == waypoint1
    assert ant1.node_solution == [waypoint1]
    assert ant1.edge_solution == []
    assert ant1.total_distance == 0
    assert ant1.total_rotation == 0

    ant1.move_to(edge_w1_w2.node2, edge_w1_w2)
    assert ant1.current_node == edge_w1_w2.node2
    assert ant1.node_solution == [waypoint1, edge_w1_w2.node2]
    assert ant1.edge_solution == [edge_w1_w2]
    assert ant1.total_distance == edge_w1_w2.length
    assert ant1.total_rotation == 0

    ant1.move_to(edge_w3_w2.node1, edge_w3_w2)
    assert ant1.current_node == waypoint3
    assert ant1.node_solution == [waypoint1, waypoint2, waypoint3]
    assert ant1.edge_solution == [edge_w1_w2, edge_w3_w2]
    assert ant1.total_distance == edge_w1_w2.length + edge_w3_w2.length
    assert ant1.total_rotation == abs(Elbow(edge_w1_w2, edge_w3_w2).angle())

    ant1.move_to(edge_w3_w4.node2, edge_w3_w4)
    assert ant1.current_node == waypoint4
    assert ant1.node_solution == [waypoint1, waypoint2, waypoint3, waypoint4]
    assert ant1.edge_solution == [edge_w1_w2, edge_w3_w2, edge_w3_w4]
    assert ant1.total_distance == edge_w1_w2.length + edge_w3_w2.length + edge_w3_w4.length
    assert ant1.total_rotation == abs(Elbow(edge_w1_w2, edge_w3_w2).angle()) + abs(Elbow(edge_w3_w2, edge_w3_w4).angle())

@pytest.fixture
def aco(graph_with_generated_edges):
    return ACO(graph_with_generated_edges.waypoints)

def test_get_optimum_path(aco):
    pass
    