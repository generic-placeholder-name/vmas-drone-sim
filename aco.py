from gettext import find
from test import Scenario
from graph import *
from typing import List
from torch import Tensor

_scenario = Scenario()

class ACO:
    def __init__(self, scenario, num_ants=50, max_iterations=100) -> None:
        self.scenario = scenario
        self.scenario.make_world(batch_dim=1, device="cpu")
        self.scenario.reset_world_at()
        self.graph = Graph(self.scenario.waypoints)
        self.graph.generate_edges()
        self.num_ants = num_ants
        self.max_iterations = max_iterations
        self.ants = []
        
    def get_optimum_path(self) -> List[Waypoint]:
        for _ in range(self.max_iterations):
            self.construct_ant_solution()
            self.update_pheromones()
        return self.graph.waypoints
            
    def construct_ant_solution(self):
        for _ in range(self.num_ants):
            ant = Ant(self.graph, self.graph.waypoints[0]) # TODO: ADD WAYPOINT RIGHT NEXT TO THEIR CHARGING STATION TO BE THERE START
            is_stuck = False
            while not (self.graph.fully_traversed() or is_stuck):
                is_stuck = not ant.move_next()
            self.ants.append(ant)

    def update_pheromones(self):
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
        self.solution = [start_node]
        self.previous_edge = None
        self.constraints = constraints # list of edges that an ant cannot traverse
        self.distance_traveled = 0
        self.radians_rotated = 0
        
    def move_next(self):
        potential_nodes, potential_paths = self.graph.get_neighbors(self.current_node, exclude_traversed=True)
        if len(potential_nodes) == 0:
            return False
        path_favorability = []
        path_heuristics = [] # [(distance, abs(angle)), ...]
        for edge in potential_paths:
            if self.previous_edge is None:
                angle = 0
            else:
                angle = Elbow(self.previous_edge, edge).angle()
            path_favorability.append(edge.pheromone * (1/edge.distance) * abs(1/(angle)))
            path_heuristics.append((edge.distance, abs(angle)))

        total_favorability = sum(path_favorability)
        probabilities = [favorability/total_favorability for favorability in path_favorability]
        next_node_index = int(Tensor.multinomial(Tensor(probabilities), 1).item())
        self.current_node = potential_nodes[next_node_index]
        self.solution.append((self.current_node, potential_paths[next_node_index]))
        self.previous_edge = potential_paths[next_node_index]

        assert path_heuristics[next_node_index][0] > 0, "Distance must be greater than 0"
        self.distance_traveled += path_heuristics[next_node_index][0]
        self.radians_rotated += path_heuristics[next_node_index][1]
    
if __name__ == "__main__":
    aco = ACO(_scenario)
    path = aco.get_optimum_path()
    print(path)