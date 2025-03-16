from vmas.simulator.core import Agent
from graph import *

class Control:
    def __init__(self, drone: Agent):
        self._drone = drone

    @property
    def drone(self):
        return self._drone
    
    def visit(self, waypoint: Waypoint):
        """
        Move the drone to the specified waypoint.
        
        Args:
            waypoint (Waypoint): The waypoint to visit.
        """
        assert isinstance(waypoint, Waypoint), "waypoint must be an instance of Waypoint"
        
        # Point heading towards waypoing

        # take a step towards waypoint