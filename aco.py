from gettext import find
from test import Scenario
from graph import *
from typing import List
from torch import Tensor

_scenario = Scenario()


class ACO:
    def __init__(self, scenario, num_ants=50, max_iterations=100, algorithm="AS", evaporation_rate=0.5, Q=1, max_pheromone=10, min_pheromone=0) -> None:
        """
        Args:
            scenario (Scenario): The scenario that the ACO algorithm will be run on
            num_ants (int): The number of ants that will be used in the algorithm
            max_iterations (int): The maximum number of iterations that the algorithm will run
            algorithm (str): The type of ACO algorithm to use. Options: AS (Ant System), MMAS (Max-Min Ant System)
            evaporation_rate (float): The rate at which pheromones evaporate
            Q (int): The pheromone base deposit, used for AS algorithm
            max_pheromone (int): The max amount of pheromone that an edge can have, used for MMAS algorithm
            min_pheromone (int): The min amount of pheromone that an edge can have, used for MMAS algorithm
        """
        self.scenario = scenario
        self.scenario.make_world(batch_dim=1, device="cpu")
        self.scenario.reset_world_at()
        self.graph = Graph(self.scenario.waypoints)
        self.graph.generate_edges()
        self.num_ants = num_ants
        self.max_iterations = max_iterations
        self.ants = []
        self.algorithm = algorithm
        # Constants
        self.evaporation_rate = evaporation_rate
        self.max_pheromone = max_pheromone
        self.min_pheromone = min_pheromone
        self.Q = Q
        
    def get_optimum_path(self) -> List[Waypoint]:
        for _ in range(self.max_iterations):
            self.graph.reset_traversed()
            self.construct_ant_solution()
            self.update_pheromones()
        return self.best_ant_solution()
    
    def best_ant_solution(self) -> List[Waypoint]:
        return self.get_best_ant().node_solution

    def get_best_ant(self):
        best_ant = None
        min_cost = float("inf")
        for ant in self.ants:
            cost = ant.distance_traveled * ant.radians_rotated
            if cost < min_cost:
                best_ant = ant
                min_cost = cost
        assert best_ant is not None, "Best ant not updated correctly"
        return best_ant

    def construct_ant_solution(self):
        for _ in range(self.num_ants):
            ant = Ant(self.graph, self.graph.waypoints[0]) # TODO: ADD WAYPOINT RIGHT NEXT TO THEIR CHARGING STATION TO BE THERE START
            is_stuck = False
            while not (self.graph.fully_traversed() or is_stuck):
                is_stuck = not ant.move_next()
            self.ants.append(ant)

    def update_pheromones(self):
        if self.algorithm == "AS":
            self.update_pheromones_as()
        elif self.algorithm == "MMAS":
            self.update_pheromones_mmas()

    def update_pheromones_as(self):
        for ant in self.ants:
            for edge in ant.edge_solution:
                added_pheromone = self.Q * heuristic(ant.distance_traveled, ant.radians_rotated)
                edge.weight = (1 - self.evaporation_rate) * edge.weight + added_pheromone

    def update_pheromones_mmas(self, ):
        pass
    
class Ant:
    def __init__(self, graph, start_node, end_node=None, constraints=[]) -> None:
        """
        Args:
            graph (Graph): The graph that the ant is traversing
            start_node (Waypoint): The node that the ant starts at
            end_node (Waypoint): The node that the ant will end at (not implemented yet)
            constraints (list): A list of edges that the ant cannot traverse
        """
        self.graph = graph
        self.current_node = start_node
        self.end_node = end_node
        self.node_solution = [start_node]
        self.edge_solution = []
        self.constraints = constraints # list of edges that an ant cannot traverse
        self.distance_traveled = 0
        self.radians_rotated = 0
        
    def move_next(self):
        potential_nodes, potential_paths = self.graph.get_neighbors(self.current_node, exclude_traversed=True)
        if len(potential_nodes) == 0:
            return False
        path_favorability = []
        path_costs = [] # [(distance, abs(angle)), ...]
        previous_edge = self.edge_solution[-1] if len(self.edge_solution) > 0 else None
        for edge in potential_paths:
            if previous_edge is None: # check if this works when edge_solution is empty
                angle = 0
            else:
                angle = Elbow(previous_edge, edge).angle()
            path_favorability.append(edge.weight * heuristic(edge.length, abs(angle)))
            path_costs.append((edge.length, abs(angle)))

        total_favorability = sum(path_favorability)
        probabilities = [favorability/total_favorability for favorability in path_favorability]
        next_node_index = int(torch.multinomial(Tensor(probabilities), 1).item())
        self.current_node = potential_nodes[next_node_index]
        self.current_node.traversed = True
        self.node_solution.append(self.current_node)
        self.edge_solution.append(potential_paths[next_node_index])

        assert path_costs[next_node_index][0] > 0, "Distance must be greater than 0"
        self.distance_traveled += path_costs[next_node_index][0]
        self.radians_rotated += path_costs[next_node_index][1]

def heuristic(distance, rotation, min_rotation_constant=0.1):
    assert distance > 0, "Distance must be greater than 0"
    assert rotation >= 0, "Rotation cannot be negative"
    distance_heuristic = 1/distance
    rotation_heuristic = 1/max(rotation, min_rotation_constant)
    return distance_heuristic * rotation_heuristic

if __name__ == "__main__":
    aco = ACO(_scenario)
    path = aco.get_optimum_path()
    print(path)