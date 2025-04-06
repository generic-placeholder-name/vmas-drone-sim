from gettext import find
from drone_simulation import Scenario
from graph import *
from typing import List
from torch import Tensor

_scenario = Scenario()
_print_log = False

class ACO:
    def __init__(self, waypoints, num_ants=20, max_iterations=100, algorithm="AS", evaporation_rate=0.001, Q=1, max_pheromone=10, min_pheromone=0, min_rotation=0.1) -> None:
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
            min_rotation (float): The minimum rotation that an ant can turn when traversing a path when determining the rotation cost. Limits favorability of an edge and prevents division by zero.
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
        self.min_rotation = min_rotation

        self.best_ant_current = None # Ant with best tour in iteration
        self.best_tour_nodes = [] # Best tour so far, specified by nodes (in all iterations)
        self.best_tour_edges = [] # Best tour so far, specified by edges (in all iterations)
        self.best_tour_distance = None
        self.best_tour_rotation = None
        
    def get_optimum_tour(self) -> List[Waypoint]:
        if _print_log:
            print(f"Running {self.algorithm} ACO with {self.num_ants} ants for {self.max_iterations} iterations")

        self.hatch_ants()
        for i in range(self.max_iterations):
            if _print_log:
                print(f"Iteration {i}")
            self.construct_ant_solution()
            self.update_best_tour()
            self.update_pheromones()

        return self.best_tour_nodes

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
            assert ant.total_distance > 0, f"Expected ant's total distance to be greater than zero. Actual: {ant.total_distance}."
            assert ant.total_rotation > 0, f"Expected ant's total rotation to be greater than zero. Actual: {ant.total_rotation}."
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
        if self.best_tour_edges == []:
            self.set_best_tour(top_ant.node_solution, top_ant.edge_solution, top_ant.total_distance, top_ant.total_rotation)
            return
        best_tour_distance, best_tour_rotation = self.graph.get_path_costs(self.best_tour_edges)
        assert best_tour_distance > 0, f"best tour distance must be greater than zero. Actual: {best_tour_distance}"
        assert best_tour_rotation > 0, f"best tour rotation must be greater than zero. Actual: {best_tour_rotation}. Num edges: {len(self.best_tour_edges)}"
        if heuristic(top_ant.total_distance, top_ant.total_rotation) > heuristic(best_tour_distance, best_tour_rotation):
            self.set_best_tour(top_ant.node_solution, top_ant.edge_solution, top_ant.total_distance, top_ant.total_rotation)

    def set_best_tour(self, tour_nodes, tour_edges, distance, rotation):
        """Set the best tour across all iterations"""
        self.best_tour_nodes = tour_nodes
        self.best_tour_edges = tour_edges
        self.best_tour_distance = distance
        assert rotation >= self.min_rotation, f"rotation cannot be below the minimum rotation to limit favorability of an edge. Rotation provided: {rotation}"
        self.best_tour_rotation = rotation

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
        for _ in range(self.num_ants):
            self.ants.append(Ant(self.graph, self.graph.waypoints[-2], self.graph.waypoints[-1], self.min_rotation)) # TODO: ADD WAYPOINT RIGHT NEXT TO THEIR CHARGING STATION TO BE THERE START

    def update_pheromones(self):
        self.evaporate()
        if self.algorithm == "AS":
            self.deposit_pheromones_as()
        elif self.algorithm == "MMAS":
            self.deposit_pheromones_mmas()
        else:
            raise ValueError(f"Expected algorithm to be 'AS' or 'MMAS'. Actual: {self.algorithm}")

    def evaporate(self):
        """Evaporation (reduction) of pheroone on all edges"""
        for edge in self.graph.edges:
            edge.weight *= (1 - self.evaporation_rate) # tau = (1 - rho) * tau

    def deposit_pheromones_as(self):
        """All ants deposit pheromones on traversed edges, following Ant System procedure"""
        for ant in self.ants:
            for edge in ant.edge_solution:
                assert ant.total_distance > 0, f"Ant's total distance must be greater than zero. Actual: {ant.total_distance}"
                assert ant.total_rotation > 0, f"Ant's total rotation must be greater than zero. Actual: {ant.total_rotation}"
                edge.add_weight(self.Q * heuristic(ant.total_distance, ant.total_rotation)) # Q * (1/L) * (1/theta)

    def deposit_pheromones_mmas(self, all_time=False):
        """The best ant deposits pheromones on traversed edges, following Max-Min Ant System"""
        assert self.best_ant_current is not None, "A best ant has not been set"
        edge_tour = None
        node_tour = None
        if all_time:
            edge_tour = self.best_tour_edges
            node_tour = self.best_tour_nodes
        else:
            edge_tour = self.best_ant_current.edge_solution
            node_tour = self.best_ant_current.node_solution

        assert edge_tour != [], "A best tour has not been declared"
        assert len(edge_tour) + 1 == len(node_tour), "Incorrect number of edges in best tour."
        distance, rotation = self.graph.get_cost_from_edges_tour(edge_tour)
        rotation = max(float(rotation), self.min_rotation) # Limit the favorability of an edge
        for edge in edge_tour:
            added_pheromone = heuristic(distance, rotation) # (1/L_best) * (1/theta_best)
            edge.weight = bound(edge.weight + added_pheromone, self.min_pheromone, self.max_pheromone) #TODO: MAKE SURE GETS HERE


class Ant:
    def __init__(self, graph, start_node, end_node=None, min_rotation=0.1) -> None:
        """
        Args:
            graph (Graph): The graph that the ant is traversing
            start_node (Waypoint): The node that the ant starts at
            end_node (Waypoint): The node that the ant will end at (not implemented yet)
            min_rotation = Minimum radians an ant can turn when traversing a path when determining cost of rotation. Required to limit favorability and avoid division by zero.
        """
        self.graph = graph
        self.start_node = start_node
        self.end_node = end_node
        self.min_rotation = min_rotation

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
        potential_nodes, potential_edges = self.graph.get_neighbors(self.current_node, exclude_traversed=True)
        if _print_log:
            print(f"Current node: {self.current_node}")
            print(f"Potential nodes: {len(potential_nodes)}")
            print(f"Potential paths: {len(potential_edges)}")
        if len(potential_nodes) == 0:
            return False
        
        # Calculate the favorability of each path
        path_favorability, path_costs = self.calculate_edges_specs(potential_edges)

        # Choose which path to take
        total_favorability = sum(path_favorability)
        probabilities = [favorability/total_favorability for favorability in path_favorability]
        next_node_index = int(torch.multinomial(Tensor(probabilities), 1).item()) # choose the next node based on the probabilities
        if _print_log:
            print(f"path favorabilities: {path_favorability}. Total: {total_favorability}")
            print(f"probabilities: {probabilities}")
            print(f"next node index: {next_node_index}")

        # Move to the next node
        self.move_to(potential_nodes[next_node_index], potential_edges[next_node_index], path_costs[next_node_index][0], path_costs[next_node_index][1])
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
            assert edge.length > 0, f"expected edge to be greater than zero. Actual: {edge.length}"
            edge_favorabilities.append(self.calculate_favorability(edge))
            edge_costs.append((edge.length, self.get_rotation(edge)))
        return edge_favorabilities, edge_costs

    def calculate_favorability(self, edge):
        """Calculate the favorability of the path based on its weight and heuristic measures"""
        return edge.weight * heuristic(edge.length, self.get_rotation(edge))
    
    def get_rotation(self, edge):
        """Get the positive angle at which the drone must rotate to get to the edge. Never trully zero."""
        previous_edge = self.edge_solution[-1] if len(self.edge_solution) > 0 else None
        return max(abs(self.graph.get_rotation(previous_edge, edge)), self.min_rotation)

def heuristic(distance, rotation):
    """Returns (1/distance) * (1/rotation), assuming some degree of rotation to prevent division by zero"""
    assert distance > 0, f"Distance must be greater than 0. Actual: {distance}"
    assert rotation > 0, f"Rotation must be greater than zero. Actual: {rotation}"
    distance_heuristic = 1/distance
    rotation_heuristic = 1/rotation
    return distance_heuristic * rotation_heuristic

def bound(value, min_bound, max_bound):
    assert min_bound < max_bound, "minimum bound must be less than maximum bound"
    return max(min(max_bound, value), min_bound)

if __name__ == "__main__":
    _scenario.make_world(batch_dim=1, device='cpu') # "cpu" underlined but doesn't cause error
    _scenario.reset_world_at()
    aco_1ant_1iteration = ACO(_scenario.waypoints, 20, 25, "MMAS")
    path = aco_1ant_1iteration.get_optimum_tour()
    if aco_1ant_1iteration.best_tour_edges is not None:
        for edge in aco_1ant_1iteration.best_tour_edges:
            print(edge)
    for waypoint in path:
        print(waypoint)

    print("\nall time best solution:\n")
    print(f"Total distance: {aco_1ant_1iteration.best_tour_distance}")
    print(f"Total radians rotated: {aco_1ant_1iteration.best_tour_rotation}")
    print(f"Performance: {heuristic(aco_1ant_1iteration.best_tour_distance, aco_1ant_1iteration.best_tour_rotation)}")
    print(f"Number of waypoints visited: {len(aco_1ant_1iteration.best_tour_nodes)} out of {len(aco_1ant_1iteration.graph.waypoints)}")

    print("\ncurrent best ant:\n")
    if aco_1ant_1iteration.best_ant_current is not None:
        print(f"Total distance: {aco_1ant_1iteration.best_ant_current.total_distance}")
        print(f"Total radians rotated: {aco_1ant_1iteration.best_ant_current.total_rotation}")
        print(f"Performance: {heuristic(aco_1ant_1iteration.best_ant_current.total_distance, aco_1ant_1iteration.best_ant_current.total_rotation)}")
        print(f"Number of waypoints visited: {len(aco_1ant_1iteration.best_ant_current.node_solution)} out of {len(aco_1ant_1iteration.graph.waypoints)}")
    
