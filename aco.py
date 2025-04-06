from gettext import find
from test import Scenario
from graph import *
from typing import List
from torch import Tensor
from test import envConfig

_scenario = Scenario()
_print_log = False


class ACO:
    def __init__(self, graph: Graph, num_ants=20, max_iterations=100, algorithm="AS", evaporation_rate=0.1, Q=1, max_pheromone=10, min_pheromone=0) -> None:
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
        self.graph = graph
        self.num_ants = num_ants
        self.max_iterations = max_iterations
        self.ants = []
        self.algorithm = algorithm
        # Constants
        self.evaporation_rate = evaporation_rate
        self.max_pheromone = max_pheromone
        self.min_pheromone = min_pheromone
        self.Q = Q

        self.best_ant = None
        
    def get_optimum_path(self) -> List[Waypoint]:
        if _print_log:
            print(f"Running {self.algorithm} ACO with {self.num_ants} ants for {self.max_iterations} iterations")

        self.hatch_ants()
        for i in range(self.max_iterations):
            if _print_log:
                print(f"Iteration {i}")
            self.construct_ant_solution()
            self.update_pheromones()
        
        self.update_best_ant()
        return self.best_ant.node_solution if self.best_ant is not None else []

    def update_best_ant(self):
        top_ant = None
        min_cost = float("inf")
        for ant in self.ants:
            cost = ant.distance_traveled * ant.radians_rotated
            if cost < min_cost:
                top_ant = ant
                min_cost = cost
        assert top_ant is not None, "Best ant not updated correctly"
        self.best_ant = top_ant

    def construct_ant_solution(self):
        """
        Constructs a solution for each ant in the ACO algorithm
        """
        for i, ant in enumerate(self.ants):
            if _print_log:
                print(f"- Ant {i}")
            self.graph.reset_traversed()
            ant.reset()
            is_stuck = False
            while not (self.graph.fully_traversed() or is_stuck):
                is_stuck = not ant.move_next()

    def hatch_ants(self):
        assert len(self.ants) == 0
        for i in range(self.num_ants):
            self.ants.append(Ant(self.graph, self.graph.waypoints[0])) # TODO: ADD WAYPOINT RIGHT NEXT TO THEIR CHARGING STATION TO BE THERE START

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

    def update_pheromones_mmas(self):
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
        self.start_node = start_node
        self.end_node = end_node
        self.constraints = constraints # list of edges that an ant cannot traverse

        self.current_node = self.start_node
        self.current_node.traversed = True
        self.node_solution = [self.start_node]
        self.edge_solution = []
        self.distance_traveled = 0
        self.radians_rotated = 0

    def reset(self):
        self.current_node = self.start_node
        self.current_node.traversed = True
        self.node_solution = [self.start_node]
        self.edge_solution = []
        self.distance_traveled = 0
        self.radians_rotated = 0
        
    def move_next(self):
        """
        Ant moves to the next node in the graph based on the pheromone levels and the heuristic function
        
        Returns:
            bool: True if the ant was able to move to the next node, False if the ant can't move
            """
        # Check if there are potential moves
        potential_nodes, potential_paths = self.graph.get_neighbors(self.current_node, exclude_traversed=True)
        if _print_log:
            print(f"Current node: {self.current_node}")
            print(f"Potential nodes: {len(potential_nodes)}")
            print(f"Potential paths: {len(potential_paths)}")
        if len(potential_nodes) == 0:
            return False
        
        # Calculate the favorability of each path
        path_favorability, path_costs = self.calculate_path_specs(potential_paths)

        # Choose which path to take
        total_favorability = sum(path_favorability)
        probabilities = [favorability/total_favorability for favorability in path_favorability]
        next_node_index = int(torch.multinomial(Tensor(probabilities), 1).item()) # choose the next node based on the probabilities
        if _print_log:
            print(f"path favorabilities: {path_favorability}. Total: {total_favorability}")
            print(f"probabilities: {probabilities}")
            print(f"next node index: {next_node_index}")

        # Move to the next node
        self.move_to(potential_nodes[next_node_index], potential_paths[next_node_index], path_costs[next_node_index][0], path_costs[next_node_index][1])
        return True
    
    def move_to(self, node, edge, distance=None, rotation=None):
        """
        Moves the ant to the specified node via the specified edge
        args:
            node (Waypoint): The node that the ant will move to
            edge (Edge): The edge that the ant will traverse
        """
        assert node in edge.nodes, "Node must be in the edge"

        # Update costs of move
        if distance is None:
            distance = edge.length
        if rotation is None:
            rotation = self.get_rotation(edge)

        assert distance > 0, "Distance must be greater than 0"
        self.distance_traveled += distance
        self.radians_rotated += rotation

        # Move to the next node
        self.current_node = node
        self.current_node.traversed = True
        self.node_solution.append(self.current_node)
        self.edge_solution.append(edge)
        if _print_log:
            print(f"Moved to {node}")
            print(f"Distance traveled: {self.distance_traveled}")
            print(f"Radians rotated: {self.radians_rotated}")

    def calculate_path_specs(self, potential_paths):
        path_favorability = []
        path_costs = [] # [(distance, abs(angle)), ...]
        for edge in potential_paths:
            rotation = self.get_rotation(edge)
            path_favorability.append(edge.weight * heuristic(edge.length, abs(rotation)))
            path_costs.append((edge.length, rotation))
        return path_favorability, path_costs
    
    def get_rotation(self, edge):
        previous_edge = self.edge_solution[-1] if len(self.edge_solution) > 0 else None
        if previous_edge is None:
            return 0
        else:
            assert previous_edge != edge, f"Can't get rotation between the same edges, {previous_edge} and {edge}"
            return abs(Elbow(previous_edge, edge).angle())

def heuristic(distance, rotation, min_rotation_constant=0.1):
    assert distance > 0, "Distance must be greater than 0"
    assert rotation >= 0, "Rotation cannot be negative"
    distance_heuristic = 1/distance
    rotation_heuristic = 1/max(rotation, min_rotation_constant)
    return distance_heuristic * rotation_heuristic

if __name__ == "__main__":
    _scenario.make_world(batch_dim=1, device='cpu') # "cpu" underlined but doesn't cause error
    _scenario.reset_world_at()
    graph = Graph(_scenario.waypoints)
    graph.generate_edges()

    penalty_areas = envConfig["penaltyAreas"] # get penalty areas from envConfig in test.py
    bad_edges = [edge for edge in graph.edges if not graph.edge_valid(edge, penalty_areas)] # edges are bad if they do not pass edge_valid
    graph.remove_edges(bad_edges) # remove bad edges

    aco = ACO(graph)
    path = aco.get_optimum_path()
    if aco.best_ant is not None:
        for edge in aco.best_ant.edge_solution:
            print(edge)
    for waypoint in path:
        print(waypoint)
    if aco.best_ant is not None:
        print(f"Total distance: {aco.best_ant.distance_traveled}")
        print(f"Total radians rotated: {aco.best_ant.radians_rotated}")
