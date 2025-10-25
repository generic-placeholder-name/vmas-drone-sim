import typing
from typing import List

import torch
from graph import *

from vmas import render_interactively
from vmas.simulator.core import Agent, Box, Landmark, Line, Sphere, World
from vmas.simulator.dynamics.diff_drive import DiffDrive
from vmas.simulator.dynamics.holonomic_with_rot import HolonomicWithRotation
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils
if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom

# Coordinates taken from image on Google Docs.
# Reward areas are areas with crops; penalty areas are the house and the greenhouse
# Original frame coordinates
original_top_left = [0, 0]
original_bottom_right = [144, 96]

original_width = original_bottom_right[0] - original_top_left[0]
original_height = original_bottom_right[1] - original_top_left[1]

# Find the center of the original region
center_x = (original_top_left[0] + original_bottom_right[0]) / 2
center_y = (original_top_left[1] + original_bottom_right[1]) / 2

# Scale based on shorter side to fit within [-1, 1]
scale = 2 / min(original_width, original_height) #2 units / 190 meters

def scale_coordinate(coord):
    """converts meters to [-1, 1] coordinates"""
    x, y = coord
    return torch.tensor([(x - center_x) * scale, (y - center_y) * scale])

def convert_to_original_units(scaled_coord):
    """Converts [-1, 1] coordinates to meters"""
    x, y = scaled_coord
    return [x / scale + center_x, y / scale + center_y]


envConfig = {
    "origBorders": { # for retrieval in ACO file
        "topLeft": original_top_left,
        "bottomRight": original_bottom_right
    },
    "borders": {
        "topLeft": scale_coordinate(original_top_left),
        "bottomRight": scale_coordinate(original_bottom_right)
    },
    # grid config: cols x rows and obstacles as (col,row) 1-based indices
    "grid": {
        "cols": 12,
        "rows": 8,
        "obstacles": [
            (2,2), (3,2), (3,7), (4,4), (5,5), (5,6), (6,5), (6,6),
            (9,3), (9,4), (9,5), (9,7), (9,8), (10,3), (10,7)
        ]
    },
    # starting points (still in scaled coords)
    "startingPoints": [
        scale_coordinate([6, 6]),
        scale_coordinate([144-6, 96-6])
    ]
}

# In polygon check (useful for distributing reward points) - left for reference
from matplotlib.path import Path
def is_point_in_polygon(point, polygon_coords):
    """
    Check if a point is inside a polygon using Matplotlib.

    Args:
        point (list or tuple): The point as a list or tuple of (x, y) coordinates.
        polygon_coords (list): The polygon as a list of (x, y) coordinate pairs.

    Returns:
        bool: True if the point is inside the polygon, False otherwise.
    """
    path = Path(polygon_coords)
    return path.contains_point(point)

class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        # Load environment configuration
        self.env_config = kwargs.pop("env_config", envConfig)
        self.shared_reward = kwargs.pop("shared_reward", False)
        self.grid_resolution = kwargs.pop("reward_grid_resolution", 0.2)
        self.agent_u_multiplier = kwargs.pop("agent_u_multiplier", 0.05)
        ScenarioUtils.check_kwargs_consumed(kwargs)

        self.n_agents = 2
        self.agent_radius = 0.03333
        self.reward_radius = 0.01
        self.visualize_semidims = True

        # Extract world dimensions from envConfig (these are scaled coords)
        world_width = self.env_config["borders"]["bottomRight"][0]
        world_height = self.env_config["borders"]["bottomRight"][1]

        self.waypoints = []
        self.obs_pos = []
        self.agent_start_pos = []
        self.last_waypoint = {i: None for i in range(self.n_agents)}
        # Make world
        world = World(batch_dim, device, x_semidim=world_width, y_semidim=world_height)
        self._world = world
        world_dims = torch.tensor([world_width, world_height])
        self.cumulative_reward = torch.zeros(
            self.world.batch_dim,
            device=self.world.device,
            dtype=torch.float32,
        )

        # Add agents
        for i in range(self.n_agents):
            agent = Agent(
                name=f"agent_{i}",
                collide=True,
                render_action=True,
                u_range=[1, 1],
                u_multiplier=[0.05, 0.5], #[linear, angular]
                shape=Sphere(self.agent_radius),
                dynamics=DiffDrive(world, integration="rk4"),
            )
            world.add_agent(agent)
            self.agent_start_pos.append(torch.tensor(self.env_config["startingPoints"][i], device=device))
        
        self.total_rotation = torch.zeros(len(self.world.agents), device=device)  # Track total rotation for each agent
        self.prev_rotations = [agent.state.rot for agent in self.world.agents]  # Track previous rotation for each agent
        print(f"World height: {world_height} \
              \nWorld width: {world_width}\n")
        
        # --- GRID-BASED WAYPOINT + OBSTACLE GENERATION ---
        grid = self.env_config["grid"]
        cols = int(grid["cols"])
        rows = int(grid["rows"])
        obstacle_cells = { (int(c), int(r)) for (c,r) in grid["obstacles"] }  # 1-based indices

        # semidims are world_width (positive). Total width = 2*world_width
        total_w = (world_width * 2).item() if isinstance(world_width, torch.Tensor) else (2 * world_width)
        total_h = (world_height * 2).item() if isinstance(world_height, torch.Tensor) else (2 * world_height)

        # cell sizes (in scaled coordinate units)
        cell_w = world_width * 2 / cols
        cell_h = world_height * 2 / rows

        # Precompute centers in scaled coords. index (col,row): col=1..cols left->right, row=1..rows bottom->top
        for col in range(1, cols + 1):
            for row in range(1, rows + 1):
                # compute center in scaled coordinates (x,y)
                # left-most center: -world_width + cell_w/2
                center_x_scaled = -world_width + (col - 0.5) * cell_w
                center_y_scaled = -world_height + (row - 0.5) * cell_h

                # If this cell is an obstacle, create obstacle landmark occupying the entire cell
                if (col, row) not in obstacle_cells:
                    # create a waypoint landmark at the cell center (these are reward points)
                    goal = Landmark(
                        name=f"goal_{len(self.waypoints)}",
                        collide=False,
                        shape=Sphere(radius=self.reward_radius),
                        color=Color.LIGHT_GREEN,
                    )
                    world.add_landmark(goal)
                    # store waypoint position in *original* units, consistent with existing code: Waypoint expects original units
                    original_units = convert_to_original_units((center_x_scaled.item(), center_y_scaled.item()))
                    self.waypoints.append(Waypoint(torch.tensor(original_units, device=device), goal, reward_radius=self.reward_radius))
                    print(f"Waypoint {len(self.waypoints)-1} created at scaled ({center_x_scaled.item()}, {center_y_scaled.item()}) -> original {original_units}")

        # waypoint_visits sized by number of waypoints generated
        self.waypoint_visits = torch.zeros([self.n_agents, len(self.waypoints)], device=device)  # Track waypoints visited by each drone
        
        # Generate waypoints at start locations 
        """
        for (x, y) in self.agent_start_pos:
            point = [x.item(), y.item()]
            goal = Landmark(
                name=f"goal_{len(self.waypoints)}",
                collide=False,
                shape=Sphere(radius=self.reward_radius),
                color=Color.LIGHT_GREEN,
            )
            world.add_landmark(goal)
            self.waypoints.append(Waypoint(torch.tensor(convert_to_original_units(point), device=device), goal, reward_radius=self.reward_radius))
            print(f"Waypoint {len(self.waypoints)-1} created at {point} = {convert_to_original_units(point)}")
        """

        # Obstacle creation (I have to do this after waypoints to keep order consistent)
        for col in range(1, cols + 1):
            for row in range(1, rows + 1):
                # compute center in scaled coordinates (x,y)
                # left-most center: -world_width + cell_w/2
                center_x_scaled = -world_width + (col - 0.5) * cell_w
                center_y_scaled = -world_height + (row - 0.5) * cell_h

                # If this cell is an obstacle, create obstacle landmark occupying the entire cell
                if (col, row) in obstacle_cells:
                    length = cell_w
                    width = cell_h
                    obstacle_shape = Box(length=length.item() if isinstance(length, torch.Tensor) else length,
                                         width=width.item() if isinstance(width, torch.Tensor) else width)
                    obstacle = Landmark(
                        name=f"obstacle_cell_{col}_{row}",
                        collide=True,
                        movable=False,
                        shape=obstacle_shape,
                        color=Color.RED,
                        collision_filter=lambda e: not isinstance(e.shape, Box),
                    )
                    world.add_landmark(obstacle)
                    # store obstacle center (scaled) for reset positioning
                    self.obs_pos.append(torch.tensor([center_x_scaled, center_y_scaled], device=device))
                    print(f"Added obstacle at cell ({col},{row}) center scaled {center_x_scaled, center_y_scaled}")

        # Note: obstacles were added earlier; obs_pos contains their scaled centers in the same order as the landmarks appended.
        # prev_positions and distance trackers
        self.prev_positions = [agent.state.pos for agent in self.world.agents]
        self.total_distance = torch.zeros(len(self.world.agents), device=device)
        
        return world
    
    def reset_world_at(self, env_index: int | None = None):
        n_goals = len(self.waypoints)
        agents = [self.world.agents[i] for i in torch.randperm(self.n_agents).tolist()]
        goals = [self.world.landmarks[i] for i in torch.range(start=0,end=n_goals-1,dtype=int).tolist()]
        order = range(len(self.world.landmarks[n_goals :]))
        obstacles = [self.world.landmarks[n_goals :][i] for i in order]
        self.waypoint_visits = torch.zeros([self.n_agents, len(self.waypoints)], device=self.world.device) # reset the counter
        self.total_distance = torch.tensor([0.0 for _ in self.world.agents])
        self.total_rotation = torch.zeros(self.n_agents, device=self.world.device)  # Reset total rotation
        self.prev_rotations = [agent.state.rot for agent in self.world.agents]  # Reset previous rotations
        for i, goal in enumerate(goals):
            goal.set_pos(
                scale_coordinate(self.waypoints[i].point),
                batch_index=env_index,
            )
        for i, agent in enumerate(agents):
            agent.set_pos(
                self.agent_start_pos[i],#self.world.agents[i].state.pos,
                batch_index=env_index,
            )
        for i, obstacle in enumerate(obstacles):
            obstacle.set_pos(
                self.obs_pos[i],
                batch_index=env_index,
            )

    def reward(self, agent: Agent):
        agent_index = self.get_agent_index(agent)
        # Track whether the agent is currently on a waypoint
        for i, landmark in enumerate(self.world.landmarks):
            if landmark.state.pos is not None and agent.state.pos is not None:
                if landmark.name.startswith("goal"):
                    if self.world.is_overlapping(agent, landmark) and self.waypoint_visits[agent_index, i] == 0:
                        waypoint_index = self.get_waypoint_index(landmark)
                        self.cumulative_reward += 1.0
                        self.waypoint_visits[agent_index, waypoint_index] += 1
                        print(f"Agent {agent_index} reached waypoint {waypoint_index}!")
                        print(f"Waypoint visits: {self.waypoint_visits[agent_index]}")
                        print(f"reward: {self.cumulative_reward}")
                        print(f"total distance: {self.total_distance[agent_index]}")
                        print("----------------------------")
                elif self.world.is_overlapping(agent, landmark):
                    if landmark.collides(agent):
                        self.cumulative_reward -= self.cumulative_reward
                        print(f"Collision by agent {agent_index}")
                        print(f"reward: {self.cumulative_reward}")
                        print("----------------------------")
                        
        #Checking drone collison, with another drone.
        for i, agent2 in enumerate(self.world.agents):
            if agent != agent2 and self.world.is_overlapping(agent, agent2):
                self.cumulative_reward -= self.cumulative_reward
                print(f"Agent {agent.name} collided with {agent2.name}!")
                print(f"reward: {self.cumulative_reward}")
                print("----------------------------")
        return self.cumulative_reward

    def observation(self, agent: Agent):
        # Update distance information
        agent_index = self.get_agent_index(agent)
        current_pos = agent.state.pos
        prev_pos = self.prev_positions[agent_index]

        # Find the distance traveled since the last step
        distance = 0.0
        if prev_pos is not None and current_pos is not None:
            distance = torch.linalg.vector_norm(current_pos - prev_pos)

        self.total_distance[agent_index] += distance
        self.prev_positions[agent_index] = current_pos

        # Update rotation information
        current_rot = agent.state.rot
        prev_rot = self.prev_rotations[agent_index]

        # Find the angular displacement since the last move
        if prev_rot is not None and current_rot is not None:

            if current_rot.dim() > 0:
                current_rot = current_rot.squeeze()  # Remove batch dimension 
            if prev_rot.dim() > 0:
                prev_rot = prev_rot.squeeze()
            angular_displacement = current_rot - prev_rot  # Gets change in rotation
            # Handle if rotation goes from 2π to 0
            angular_displacement = (angular_displacement + torch.pi) % (2 * torch.pi) - torch.pi

            # Add absolute value angular displacement
            self.total_rotation[agent_index] += torch.abs(angular_displacement)


        self.prev_rotations[agent_index] = current_rot

        # Get positions of all landmarks in this agent's reference frame
        landmark_rel_poses = []
        for landmark in self.world.landmarks:
            assert landmark.state.pos is not None and agent.state.pos is not None, "Landmark or agent position is None"
            landmark_rel_poses.append(landmark.state.pos - agent.state.pos)
        return torch.cat(
            [
                agent.state.pos if agent.state.pos is not None else torch.zeros(2, device=agent.device),
                agent.state.vel if agent.state.vel is not None else torch.zeros(2, device=agent.device),
                *landmark_rel_poses,
            ],
            dim=-1,
        )
    
    def get_agent_index(self, agent: Agent):
        return int(agent.name.split("_")[1])
    
    def get_waypoint_index(self, goal: Landmark):
        return int(goal.name.split("_")[1])

    # def done(self): not implemented yet

    # def extra_render(self, env_index: int = 0):
    def extra_render(self, env_index: int = 0) -> "List[Geom]":

        geoms: List[Geom] = []

        # Agent rotation
        for agent in self.world.agents:
            geoms.append(
                ScenarioUtils.plot_entity_rotation(agent, env_index, length=0.1)
            )

        return geoms

if __name__ == "__main__":
    render_interactively(
        Scenario(), control_two_agents=True, shared_reward=True
    )
