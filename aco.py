from gettext import find
from test import Scenario
from graph import *

_scenario = Scenario()

class ACO:
    def __init__(self, scenario) -> None:
        self.scenario = scenario
        self.scenario.make_world(batch_dim=1, device="cpu")
        self.scenario.reset_world_at()
        self.graph = None

    def generate_graph(self):
        self.graph = Graph(self.scenario.waypoints)
        self.graph.generate_edges()

    


if __name__ == "__main__":
    aco = ACO(_scenario)
    aco.generate_graph()
    print(aco.graph)