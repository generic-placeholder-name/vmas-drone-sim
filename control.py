from math import isclose
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
        assert self.drone.state.pos is not None, "drone state position must be initialized"
        assert self.drone.state.rot is not None, "drone state rotation must be initialized"
        
        # Check if the drone is within the reward radius of the waypoint
        if torch.linalg.vector_norm(self.drone.state.pos - waypoint.point) < waypoint.reward_radius:
            # If within the reward radius, stop the drone
            self.drone.dynamics.agent.action.u[:, 0] = 0
            self.drone.dynamics.agent.action.u[:, 1] = 0

        # Point heading towards waypoing
        if self.drone.state.rot < (waypoint.point - self.drone.state.pos).angle():
            self.drone.dynamics.agent.action.u[:, 1] = 1.0 # right angular velocity
        else:
            self.drone.dynamics.agent.action.u[:, 1] = -1.0 # left angular velocity

        # If the drone is facing the waypoint, set forward velocity
        if torch.isclose(self.drone.state.rot, (waypoint.point - self.drone.state.pos).angle()):
            self.drone.dynamics.agent.action.u[:, 0] = 1 # Move forwards
        else:
            self.drone.dynamics.agent.action.u[:, 0] = 0 # Stop moving forward
        
        self.drone.dynamics.process_action()