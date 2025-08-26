from gettext import find
from drone_simulation import Scenario
from graph import *
from typing import List
from torch import Tensor
from drone_simulation import envConfig, droneRadius, dronePositionUncertainty
from draw import draw_paths
import optuna
import csv

_scenario = Scenario()
_print_log = False

class ACO:
    def __init__(self, graph: Graph, start: Waypoint, num_ants=414, max_iterations=138, algorithm="MMAS", evaporation_rate=0.01336, Q=1, min_pheromone=1.0, max_pheromone=2.027, lambda_=0.1164, gamma=0.0173, alpha=3.985, beta=2.391) -> None:
        """
        Args:
            graph (Graph): The graph containing all of the waypoints that should be traversed
            start (Waypoint): The node that the ants start and end at
            num_ants (int): The number of ants that will be used in the algorithm
            max_iterations (int): The maximum number of iterations that the algorithm will run
            algorithm (str): The type of ACO algorithm to use. Options: AS (Ant System), MMAS (Max-Min Ant System)
            evaporation_rate (float): The rate at which pheromones evaporate
            Q (int): The pheromone base deposit, used for AS algorithm
            min_pheromone (float): The min amount of pheromone that an edge can have, used for MMAS algorithm
            max_pheromone (float): The max amount of pheromone that an edge can have, used for MMAS algorithm
            lambda_ (float): The cost of traveling in a straight line given in kJ/m
            gamma (float): The cost of rotating given in kJ/degree
            alpha (float): The importance of the pheromone in the favorability calculation
            beta (float): The importance of the heuristic measure in the favorability calculation
        """
        self.graph = graph
        self.start = start
        self.num_ants = num_ants
        self.max_iterations = max_iterations
        self.ants = []
        self.algorithm = algorithm
        # Constants
        self.evaporation_rate = evaporation_rate
        self.min_pheromone = min_pheromone
        self.max_pheromone = max_pheromone
        self.Q = Q
        self.lambda_ = lambda_
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta

        self.best_ant_current = None # Ant with best tour in iteration
        self.best_tour_nodes = [] # Best tour so far, specified by nodes (in all iterations)
        self.best_tour_edges = [] # Best tour so far, specified by edges (in all iterations)
        self.best_tour_distance = None
        self.best_tour_rotation = None

        assert self.start in self.graph.waypoints, f"Start node must be in the graph. Start node: {self.start}. Graph waypoints: {self.graph.waypoints}"
        self.initialize_pheromones()

    def get_optimum_tour(self) -> List[Waypoint]:
        """Return the best tour found by the ACO algorithm"""
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
    
    def initialize_pheromones(self):
        """Initialize the pheromones on the edges of the graph"""
        for edge in self.graph.edges:
            edge.weight = self.max_pheromone

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
            performance = heuristic(ant.total_distance, ant.total_rotation, self.lambda_, self.gamma)
            if performance > best_performance:
                top_ant = ant
                num_waypoints_toured = len(ant.node_solution)
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
        assert best_tour_rotation >= 0, f"best tour rotation must be zero or greater. Actual: {best_tour_rotation}. Num edges: {len(self.best_tour_edges)}"
        if heuristic(top_ant.total_distance, top_ant.total_rotation, self.lambda_, self.gamma) > heuristic(best_tour_distance, best_tour_rotation, self.lambda_, self.gamma):
            self.set_best_tour(top_ant.node_solution, top_ant.edge_solution, top_ant.total_distance, top_ant.total_rotation)

    def set_best_tour(self, tour_nodes, tour_edges, distance, rotation):
        """Set the best tour across all iterations"""
        self.best_tour_nodes = tour_nodes
        self.best_tour_edges = tour_edges
        self.best_tour_distance = distance
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
            # There are no more waypoints that the ant can visit (either all are traversed or there are no valid edges)
            self.start.traversed = False # Give the ant the option to return to the start node
            ant.move_next() # Move to the start node

    def hatch_ants(self):
        """Instantiate the ants"""
        assert len(self.ants) == 0
        for _ in range(self.num_ants):
            self.ants.append(Ant(self.graph, self.start, self.lambda_, self.gamma, self.alpha, self.beta))

    def update_pheromones(self):
        """Update the pheromones on the edges of the graph based on the algorithm selected"""
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
            edge.weight = bound(edge.weight * (1 - self.evaporation_rate), self.min_pheromone, self.max_pheromone) # tau = (1 - rho) * tau

    def deposit_pheromones_as(self):
        """All ants deposit pheromones on traversed edges, following Ant System procedure"""
        for ant in self.ants:
            for edge in ant.edge_solution:
                assert ant.total_distance > 0, f"Ant's total distance must be greater than zero. Actual: {ant.total_distance}"
                assert ant.total_rotation > 0, f"Ant's total rotation must be greater than zero. Actual: {ant.total_rotation}"
                edge.add_weight(self.Q * heuristic(ant.total_distance, ant.total_rotation, self.lambda_, self.gamma)) # Q * (1/L) * (1/theta)

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
        for edge in edge_tour:
            added_pheromone = heuristic(distance, rotation, self.lambda_, self.gamma) # 1/(lambda * L_best + gamma * theta_best)
            edge.weight = bound(edge.weight + added_pheromone, self.min_pheromone, self.max_pheromone)


class Ant:
    def __init__(self, graph, start_node, lambda_, gamma, alpha, beta) -> None:
        """
        Args:
            graph (Graph): The graph that the ant is traversing
            start_node (Waypoint): The node that the ant starts at
            lambda_ (float): The cost of traveling in a straight line given in kJ/m
            gamma (float): The cost of rotating given in kJ/degree
            alpha (float): The importance of the pheromone in the favorability calculation
            beta (float): The importance of the heuristic measure in the favorability calculation
        """
        self.graph = graph
        self.start_node = start_node
        self.lambda_ = lambda_
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta

        self.current_node = self.start_node
        self.current_node.traversed = True
        self.node_solution = [self.start_node]
        self.edge_solution = []
        self.total_distance = 0
        self.total_rotation = 0 # in degrees

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
        """Calculate the favorability of the path based on its heuristic measure and pheromone"""
        return  heuristic(edge.length, self.get_rotation(edge), self.lambda_, self.gamma)**self.alpha * edge.weight**self.beta
    
    def get_rotation(self, edge):
        """Get the positive angle at which the drone must rotate to get to the edge. Never trully zero."""
        previous_edge = self.edge_solution[-1] if len(self.edge_solution) > 0 else None
        return abs(self.graph.get_rotation(previous_edge, edge))
    
def cost(distance, rotation, lambda_=0.1164, gamma=0.0173):
    """
    Cost function for the ACO algorithm. This function is used to calculate the cost of a path based on its distance and rotation.
    Args:
        distance (float): The distance of the path
        rotation (float): The rotation of the path
        lambda_ (float): The cost of traveling in a straight given in kJ/m
        gamma (float): The cost of rotating given in kJ/degree
    Returns:
        float: The cost of the path
    """
    assert distance > 0, f"Distance must be greater than 0. Actual: {distance}"
    assert rotation >= 0, f"Rotation must be zero or greater. Actual: {rotation}"
    return lambda_ * distance + gamma * rotation

def heuristic(distance, rotation, lambda_=0.1164, gamma=0.0173):
    """Returns favorability of a tour, or 1/(lambda * distance + gamma * rotation)"""
    assert distance > 0, f"Distance must be greater than 0. Actual: {distance}"
    assert rotation >= 0, f"Rotation must be zero or greater. Actual: {rotation}"
    return 1 / cost(distance, rotation, lambda_, gamma)

def bound(value, min_bound, max_bound):
    """Returns the value bounded by the min and max bounds"""
    assert min_bound < max_bound, "minimum bound must be less than maximum bound"
    return max(min(max_bound, value), min_bound)

def solve_eecpp_problem(graph: Graph, start: Waypoint, num_ants=414, max_iterations=138, algorithm="MMAS", evaporation_rate=0.01336, Q=1, min_pheromone=1.0, max_pheromone=2.027, lambda_=0.1164, gamma=0.0173, alpha=3.985, beta=2.391):
    """Solve the Energy Efficient Coverage Path Planning problem using the ACO algorithm."""
    aco = ACO(graph, start, num_ants, max_iterations, algorithm, evaporation_rate, Q, min_pheromone, max_pheromone, lambda_, gamma, alpha, beta)
    best_tour = aco.get_optimum_tour()
    return best_tour, aco

  
def draw_aco_solution(tour1, tour2=[], file_name="coverage_path.png"):
    """Draw solution generated by ACO"""
    print("\nDrawing paths...\n")
    # Draw the graph with the best tours
    all_wps = tour1 + tour2
    paths = [[Edge(best_tour[i], best_tour[i+1]) for i in range(len(best_tour)-1)] for best_tour in (tour1, tour2)]
    print('*' * 20)
    for edge in paths[0]:
        print(edge.node1.point, edge.node2.point)
    print('*' * 20)
    for edge in paths[1]:
            print(edge.node1.point, edge.node2.point)
    print('*' * 20)
    dims = envConfig["origBorders"]["topLeft"] + envConfig["origBorders"]["bottomRight"]
    draw_paths(
        waypoints=all_wps,
        paths=paths,
        obstacles=envConfig["penaltyAreas"],
        dimensions=dims,
        file_path=file_name,
    )

def objective(trial):
    """Objective function for the optuna optimization"""
    algorithm = "MMAS"

    # Parameter search spacce
    n = trial.suggest_int("n", 10, 500) # number of ants
    iter = trial.suggest_int("iter", 10, 200) # number of iterations
    rho = trial.suggest_float("rho", 0.001, 1) # evaporation rate
    alpha = trial.suggest_float("alpha", 0.1, 5) # heuristic importance
    beta = trial.suggest_float("beta", 0.1, 5) # pheromone importance
    Q = 1
    tau_min = 1.0 # minimum pheromone an edge can have
    tau_max = trial.suggest_float("tau_max", 2, 10) # maximum pheromone an edge can have
    lambda_ = 0.1164 # cost of traveling in a straight line given in kJ/m (constant)
    gamma = 0.0173 # cost of rotating given in kJ/degree (constant)
    
    #Find tours for each drone around the farm
    _scenario.make_world(batch_dim=1, device='cpu') # "cpu" underlined but doesn't cause error
    _scenario.reset_world_at()

    penalty_areas = envConfig["penaltyAreas"] # get penalty areas from envConfig in drone_simulaion.py
    tour1, aco1 = solve_eecpp_problem([_scenario.waypoints[-1]] + _scenario.waypoints[:15], _scenario.waypoints[-1], penalty_areas, num_ants=n, max_iterations=iter, algorithm=algorithm, evaporation_rate=rho, Q=Q, min_pheromone=tau_min, max_pheromone=tau_max, lambda_=lambda_, gamma=gamma, alpha=alpha, beta=beta)
    tour2, aco2 = solve_eecpp_problem([_scenario.waypoints[-2]] + _scenario.waypoints[15:-2], _scenario.waypoints[-2], penalty_areas, num_ants=n, max_iterations=iter, algorithm=algorithm, evaporation_rate=rho, Q=Q, min_pheromone=tau_min, max_pheromone=tau_max, lambda_=lambda_, gamma=gamma, alpha=alpha, beta=beta)
    if aco1.best_tour_nodes[0] != aco1.best_tour_nodes[-1] or aco2.best_tour_nodes[0] != aco2.best_tour_nodes[-1]:
        # At least one drone does not return to the charging station
        return 0
    
    # Return performance of the tours. Higher number is better as it means less energy cost.
    return heuristic(aco1.best_tour_distance, aco1.best_tour_rotation, lambda_, gamma) + heuristic(aco2.best_tour_distance, aco2.best_tour_rotation, lambda_, gamma)

def compare_aco_algorithms_2drones(num_trials=10):
    """Generate file of performance of AS vs MMAS"""
    data = [["Algorithm", "Distance", "Rotation", "Energy Cost", "Num Waypoints", "Returned to Start"]]
    
    _scenario.make_world(batch_dim=1, device='cpu') # "cpu" underlined but doesn't cause error
    _scenario.reset_world_at()
    
    penalty_areas = envConfig["penaltyAreas"] # get penalty areas from envConfig in test.py
    graph1 = prepare_graph([_scenario.waypoints[-2]] + _scenario.waypoints[:15], penalty_areas)
    graph2 = prepare_graph([_scenario.waypoints[-1]] + _scenario.waypoints[15:-2], penalty_areas)
    
    for i in range(num_trials):
        graph1.reset()
        _, aco1_as = solve_eecpp_problem(graph1, _scenario.waypoints[-2], algorithm="AS")
        assert aco1_as.best_tour_distance is not None and aco1_as.best_tour_rotation is not None
        assert aco1_as.best_tour_nodes is not None

        graph2.reset()
        _, aco2_as = solve_eecpp_problem(graph2, _scenario.waypoints[-1], algorithm="AS")
        assert aco2_as.best_tour_distance is not None and aco2_as.best_tour_rotation is not None
        assert aco2_as.best_tour_nodes is not None
        data.append(process_aco_results(aco1_as, aco2_as))
        
    for i in range(num_trials):
        graph1.reset()
        _, aco1_as = solve_eecpp_problem(graph1, _scenario.waypoints[-2], algorithm="MMAS")
        assert aco1_as.best_tour_distance is not None and aco1_as.best_tour_rotation is not None
        assert aco1_as.best_tour_nodes is not None

        graph2.reset()
        _, aco2_as = solve_eecpp_problem(graph2, _scenario.waypoints[-1], algorithm="MMAS")
        assert aco2_as.best_tour_distance is not None and aco2_as.best_tour_rotation is not None
        assert aco2_as.best_tour_nodes is not None
        data.append(process_aco_results(aco1_as, aco2_as))

    filename = f'as_mmas_2drone_performance_{num_trials}_trials.csv'
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
    print(f'Generated aco performance comparison file: {filename}')

def compare_aco_algorithms_1drone(num_trials=30):
    """Generate file of performance of AS vs MMAS"""
    data = [["Algorithm", "Distance", "Rotation", "Energy Cost", "Num Waypoints", "Returned to Start"]]
    
    _scenario.make_world(batch_dim=1, device='cpu') # "cpu" underlined but doesn't cause error
    _scenario.reset_world_at()
    
    penalty_areas = envConfig["penaltyAreas"] # get penalty areas from envConfig in test.py
    graph = prepare_graph([_scenario.waypoints[-2]] + _scenario.waypoints[:-2], penalty_areas)
    
    for i in range(num_trials):
        graph.reset()
        _, aco_as = solve_eecpp_problem(graph, _scenario.waypoints[-2], algorithm="AS")
        assert aco_as.best_tour_distance is not None and aco_as.best_tour_rotation is not None
        assert aco_as.best_tour_nodes is not None
        data.append([aco_as.algorithm,
                     aco_as.best_tour_distance.item(),
                     aco_as.best_tour_rotation.item(),
                     cost(aco_as.best_tour_distance, aco_as.best_tour_rotation).item(),
                     str(len(aco_as.best_tour_nodes)),
                     aco_as.best_tour_nodes[0] == aco_as.best_tour_nodes[-1]])
        
    for i in range(num_trials):
        graph.reset()
        _, aco_as = solve_eecpp_problem(graph, _scenario.waypoints[-2], algorithm="MMAS")
        assert aco_as.best_tour_distance is not None and aco_as.best_tour_rotation is not None
        assert aco_as.best_tour_nodes is not None
        data.append([aco_as.algorithm,
                     aco_as.best_tour_distance.item(),
                     aco_as.best_tour_rotation.item(),
                     cost(aco_as.best_tour_distance, aco_as.best_tour_rotation).item(),
                     str(len(aco_as.best_tour_nodes)),
                     aco_as.best_tour_nodes[0] == aco_as.best_tour_nodes[-1]])

    filename = f'as_mmas_1drone_performance_{num_trials}_trials.csv'
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
    print(f'Generated aco performance comparison file: {filename}')

def process_aco_results(aco1, aco2):
    """Combine results of the aco algorithms for both drones"""
    aco_distance = aco1.best_tour_distance + aco2.best_tour_distance
    aco_rotation = aco1.best_tour_rotation + aco2.best_tour_rotation
    assert aco_distance > 0, f"Expected ACO AS distance to be greater than zero. Actual: {aco_distance}"
    assert aco_rotation > 0, f"Expected ACO AS rotation to be greater than zero. Actual: {aco_rotation}"
    aco_cost = cost(aco_distance, aco_rotation)
    aco_as_number_of_waypoints = len(aco1.best_tour_nodes) + len(aco2.best_tour_nodes)
    aco_as_returned_to_start = aco1.best_tour_nodes[0] == aco1.best_tour_nodes[-1] and aco2.best_tour_nodes[0] == aco2.best_tour_nodes[-1]
    assert aco1.algorithm == aco2.algorithm, f"Expected both drones to use the same ACO algorithm. Actual: {aco1.algorithm} and {aco2.algorithm}"
    return [aco1.algorithm,
            aco_distance.item(),
            aco_rotation.item(),
            aco_cost.item(),
            str(aco_as_number_of_waypoints),
            aco_as_returned_to_start]


def prepare_graph(waypoints: List[Waypoint], penalty_areas=[], collision_margin=None) -> Graph:
    if collision_margin == None:
        collision_margin = droneRadius + dronePositionUncertainty
    graph = Graph(waypoints, margin=collision_margin)
    graph.generate_edges(penalty_areas, generate_alternative_routes=True)
    return graph
    
def optimize_aco():
    """Find best performing hyperparameters for the ACO algorithm on the farm"""
    # Create and run the study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=40) # change too 100 later

    # Output best parameters
    print("Best parameters:", study.best_params)
    print("Best value:", study.best_value)

    # Save results if needed
    df = study.trials_dataframe()
    df.to_csv("optimization_results.csv", index=False)

def generate_1drone_aco_path(alg="MMAS", ants=414):
    _scenario.make_world(batch_dim=1, device='cpu') # "cpu" underlined but doesn't cause error
    _scenario.reset_world_at()

    penalty_areas = envConfig["penaltyAreas"] # get penalty areas from envConfig in test.py
    graph = prepare_graph(_scenario.waypoints[:-1], penalty_areas)
    tour, aco = solve_eecpp_problem(graph, _scenario.waypoints[-2], num_ants=ants, algorithm=alg)

    print("\ntour:\n")
    for waypoint in tour:
        print(waypoint)

    print("\nfirst tour costs\n")
    print(f"Total distance: {aco.best_tour_distance}")
    print(f"Total degrees rotated: {aco.best_tour_rotation}")
    print(f"Performance: {heuristic(aco.best_tour_distance, aco.best_tour_rotation)}")
    print(f"Number of waypoints visited: {len(aco.best_tour_nodes[1:])} out of {len(aco.graph.waypoints)}")
    print(f"Returned to start node: {aco.best_tour_nodes[0] == aco.best_tour_nodes[-1]}")
    
    draw_aco_solution(tour, file_name=f"aco_{alg}_1drone_path.png")


if __name__ == "__main__":
    generate_1drone_aco_path(alg="MMAS", ants=100)

    # Uncomment to compare AS and MMAS performance
    # compare_aco_algorithms_1drone(30)

    # Uncomment to run hyperparameter tuning
    #optimize_aco()

    # Uncomment to find tours for each drone around the farm
    # _scenario.make_world(batch_dim=1, device='cpu') # "cpu" underlined but doesn't cause error
    # _scenario.reset_world_at()

    # penalty_areas = envConfig["penaltyAreas"] # get penalty areas from envConfig in test.py
    # graph1 = prepare_graph([_scenario.waypoints[-2]] + _scenario.waypoints[:15], penalty_areas)
    # graph2 = prepare_graph([_scenario.waypoints[-1]] + _scenario.waypoints[15:-2], penalty_areas)

    # tour1, aco1 = solve_eecpp_problem(graph1, _scenario.waypoints[-2], algorithm="MMAS")
    # tour2, aco2 = solve_eecpp_problem(graph2, _scenario.waypoints[-1], algorithm="MMAS")

    # print("\nfirst tour:\n")
    # for waypoint in tour1:
    #     print(waypoint)
    
    # print("\nsecond tour:\n")
    # for waypoint in tour2:
    #     print(waypoint)

    # print("\nfirst tour costs\n")
    # print(f"Total distance: {aco1.best_tour_distance}")
    # print(f"Total degrees rotated: {aco1.best_tour_rotation}")
    # print(f"Performance: {heuristic(aco1.best_tour_distance, aco1.best_tour_rotation)}")
    # print(f"Number of waypoints visited: {len(aco1.best_tour_nodes[1:])} out of {len(aco1.graph.waypoints)}")
    # print(f"Returned to start node: {aco1.best_tour_nodes[0] == aco1.best_tour_nodes[-1]}")

    # print("\nsecond tour costs\n")
    # print(f"Total distance: {aco2.best_tour_distance}")
    # print(f"Total degrees rotated: {aco2.best_tour_rotation}")
    # print(f"Performance: {heuristic(aco2.best_tour_distance, aco2.best_tour_rotation)}")
    # print(f"Number of waypoints visited: {len(aco2.best_tour_nodes[1:])} out of {len(aco2.graph.waypoints)}")
    # print(f"Returned to start node: {aco2.best_tour_nodes[0] == aco2.best_tour_nodes[-1]}")
    
    # draw_aco_solution(tour1, tour2)
