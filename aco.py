from gettext import find
from drone_simmulation import Scenario
from graph import *
from typing import List
from torch import Tensor

_scenario = Scenario()
_print_log = False

class ACO:
    def __init__(self, waypoints, num_ants=20, max_iterations=100, algorithm="AS", evaporation_rate=0.1, Q=1, max_pheromone=10, min_pheromone=0) -> None:
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
        self.graph = Graph(waypoints)
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

        self.best_ant_current = None # Ant with best tour in iteration
        self.best_tour_nodes = [] # Best tour so far, specified by nodes (in all iterations)
        self.best_tour_edges = [] # Best tour so far, specified by edges (in all iterations)
        
    def get_optimum_tour(self) -> List[Waypoint]:
        if _print_log:
            print(f"Running {self.algorithm} ACO with {self.num_ants} ants for {self.max_iterations} iterations")

        self.hatch_ants()
        for i in range(self.max_iterations):
            if _print_log:
                print(f"Iteration {i}")
            self.construct_ant_solution()
            self.update_pheromones()
        
        self.update_best_tour()
        return self.best_ant_current.node_solution if self.best_ant_current is not None else []

    def update_best_tour(self):
        """Update the best tours across all iterations, and also the best ant in the latest iteration"""
        top_ant = None
        num_waypoints_toured = 0
        best_performance = 0
        for ant in self.ants:
            # Do not include shorter tours
            if len(ant.node_solution) < num_waypoints_toured:
                continue
            
            # Check if ant is top performing in this iteration
            performance = heuristic(ant.total_distance, ant.total_rotation)
            if performance > best_performance:
                top_ant = ant
                best_performance = performance

        assert top_ant is not None, "Best ant not updated correctly"
        self.best_ant_current = top_ant

        # Don't update best tour if previous best was longer (hit more waypoints)
        if len(self.best_ant_current.node_solution) < len(self.best_tour_nodes):
            return
        
        # Update best tours if best ant this iteration was the best-so-far
        if self.best_tour_edges is None:
            self.set_best_tour(top_ant.node_solution, top_ant.edge_solution)
            return
        best_tour_distance, best_tour_rotation = self.graph.get_path_costs(self.best_tour_edges)
        if heuristic(top_ant.total_distance, top_ant.total_rotation) > heuristic(best_tour_distance, best_tour_rotation):
            self.set_best_tour(top_ant.node_solution, top_ant.edge_solution)

    def set_best_tour(self, tour_nodes, tour_edges):
        """Set the best tour across all iterations"""
        self.best_tour_nodes = tour_nodes
        self.best_tour_edges = tour_edges

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
        self.evaporate()
        if self.algorithm == "AS":
            self.update_pheromones_as()
        elif self.algorithm == "MMAS":
            self.update_pheromones_mmas()

    def evaporate(self):
        """Evaporation (reduction) of pheroone on all edges"""
        for edge in self.graph.edges:
            edge.weight *= (1 - self.evaporation_rate) # tau = (1 - rho) * tau

    def update_pheromones_as(self):
        """All ants deposit pheromones on traversed edges, following Ant System procedure"""
        for ant in self.ants:
            for edge in ant.edge_solution:
                edge.weight += self.Q * heuristic(ant.total_distance, ant.total_rotation) # Q * (1/L) * (1/theta)

    def update_pheromones_mmas(self):
        """The best ant deposits pheromones on traversed edges, following Max-Min Ant System"""
        assert self.best_ant_current is not None, "A best ant has not been declared" #TODO: Make sure each iteration one is declared
        for edge in self.best_ant_current.edge_solution:
            added_pheromone = heuristic(self.best_ant_current.total_distance, self.best_ant_current.total_rotation) # (1/L_best) * (1/theta_best)
            edge.weight = bound(edge.weight + added_pheromone, self.min_pheromone, self.max_pheromone)
    

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
        self.total_distance = 0
        self.total_rotation = 0 # in radians

    def reset(self):
        self.current_node = self.start_node
        self.current_node.traversed = True
        self.node_solution = [self.start_node]
        self.edge_solution = []
        self.total_distance = 0
        self.total_rotation = 0
        
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
        path_favorability, path_costs = self.calculate_edges_specs(potential_paths)

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
        breakpoint()
        assert distance > 0, "Distance must be greater than 0"
        self.total_distance += distance
        self.total_rotation += rotation

        # Move to the next node
        self.current_node = node
        self.current_node.traversed = True
        self.node_solution.append(self.current_node)
        self.edge_solution.append(edge)
        if _print_log:
            print(f"Moved to {node}")
            print(f"Distance traveled: {self.total_distance}")
            print(f"Radians rotated: {self.total_rotation}")

    def calculate_edges_specs(self, potential_edges):
        """
        Get the favorability values and costs for each edge provided.
        """
        edge_favorabilities = []
        edge_costs = [] # [(distance, abs(angle)), ...]
        for edge in potential_edges:
            edge_favorabilities.append(self.calculate_favorability(edge))
            edge_costs.append((edge.length, self.get_rotation(edge)))
        return edge_favorabilities, edge_costs

    def calculate_favorability(self, edge):
        return edge.weight * heuristic(edge.length, abs(self.get_rotation(edge)))
    
    def get_rotation(self, edge):
        previous_edge = self.edge_solution[-1] if len(self.edge_solution) > 0 else None
        return self.graph.get_rotation(previous_edge, edge)


def heuristic(distance, rotation, min_rotation_constant=0.1):
    """Returns (1/distance) * (1/rotation), assuming some degree of rotation to prevent division by zero"""
    assert distance > 0, "Distance must be greater than 0"
    assert rotation >= 0, "Rotation cannot be negative"
    distance_heuristic = 1/distance
    rotation_heuristic = 1/max(rotation, min_rotation_constant)
    return distance_heuristic * rotation_heuristic

def bound(value, min_bound, max_bound):
    max(min(max_bound, edge.weight + value), min_bound)

if __name__ == "__main__":
    _scenario.make_world(batch_dim=1, device='cpu') # "cpu" underlined but doesn't cause error
    _scenario.reset_world_at()
    aco = ACO(_scenario.waypoints, 5, 1)
    path = aco.get_optimum_tour()
    if aco.best_ant_current is not None:
        for edge in aco.best_ant_current.edge_solution:
            print(edge)
    for waypoint in path:
        print(waypoint)
    if aco.best_ant_current is not None:
        print(f"Total distance: {aco.best_ant_current.total_distance}")
        print(f"Total radians rotated: {aco.best_ant_current.total_rotation}")
