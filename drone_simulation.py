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
# original_top_left = [181, 7]
# original_bottom_right = [486, 253]
# Changed to meters x meters of farm, measured from google maps
original_top_left = [0, 0]
original_bottom_right = [245, 190]

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
    "rewardAreas": [
        [scale_coordinate([0, 0]), 
         scale_coordinate([original_width, 0]), 
         scale_coordinate([original_width, original_height]), 
         scale_coordinate([0, original_height])]
        # [scale_coordinate(coord) for coord in [
        #     [187, 130], [241, 153], [265, 153], [261, 240], [184, 239]
        # ]],
        # [scale_coordinate(coord) for coord in [
        #     [213, 58], [229, 57], [244, 77], [274, 78], [305, 67], [293, 140], [255, 140], [204, 120]
        # ]],
        # [scale_coordinate(coord) for coord in [
        #     [311, 58], [360, 60], [360, 140], [303, 140]
        # ]],
        # [scale_coordinate(coord) for coord in [
        #     [365, 16], [471, 15], [475, 87], [364, 89]
        # ]],
        # [scale_coordinate(coord) for coord in [
        #     [372, 173], [477, 174], [480, 240], [370, 241]
        # ]],
        # [scale_coordinate(coord) for coord in [
        #     [381, 98], [476, 92], [479, 166], [382, 164]
        # ]]
    ],
    "penaltyAreas": [ 
        {
            #Second coordinate is subtracted because  value, i.e. 500, was measured from top-left of image, vmas wants it from bottom left
            "topLeft": [107, original_height-152],  
            "bottomRight": [130, original_height-122],
            "type": "box"
        },
        {
            "topLeft": [148, original_height-104],
            "bottomRight": [160, original_height-74],
            "type": "box"
        },
        {
            "topLeft": [107, original_height-183],
            "bottomRight": [122, original_height-168],
            "type": "circle",
        },
        {
            "topLeft": [53, original_height-61],
            "bottomRight": [69, original_height-46],
            "type": "circle",
        }
    ],
    "startingPoints": [
        scale_coordinate([137, 61]),
        scale_coordinate([145, 61])
    ]
}

# In polygon check (useful for distributing reward points)
# Uses matplotlib (which is probably not the best)
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

        # Extract world dimensions from envConfig
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
        
        # Generate goal (waypoints) points in reward areas
        for x in torch.arange(-world_width + self.grid_resolution, world_width, self.grid_resolution*2):
            for y in torch.arange(-world_height + self.grid_resolution, world_height, self.grid_resolution*2):
                point = [x.item(), y.item()]
                for reward_area in self.env_config["rewardAreas"]:
                    if is_point_in_polygon(point, reward_area): # TODO: Check that point not in penalty areas
                        print("Is in reward area\n")
                        goal = Landmark(
                            name=f"goal_{len(self.waypoints)}",
                            collide=False,
                            shape=Sphere(radius=self.reward_radius),
                            color=Color.LIGHT_GREEN,
                        )
                        # if agent in point
                        world.add_landmark(goal)
                        self.waypoints.append(Waypoint(torch.tensor(convert_to_original_units(point), device=device), goal, reward_radius=self.reward_radius))
                        print(f"Waypoint {len(self.waypoints)-1} created at {point} = {convert_to_original_units(point)}")
                        break

        # Generate waypoints at start locations
        for (x, y) in self.agent_start_pos:
            point = [x.item(), y.item()]
            goal = Landmark(
                name=f"goal_{len(self.waypoints)}",
                collide=False,
                shape=Sphere(radius=self.reward_radius),
                color=Color.LIGHT_GREEN,
            )
            # if agent in point
            world.add_landmark(goal)
            self.waypoints.append(Waypoint(torch.tensor(convert_to_original_units(point), device=device), goal, reward_radius=self.reward_radius))
            print(f"Waypoint {len(self.waypoints)-1} created at {point} = {convert_to_original_units(point)}")

        self.waypoint_visits = torch.zeros([self.n_agents, len(self.waypoints)], device=device)  # Track waypoints visited by each drone
        
        # Add penalty areas as landmarks
        for i, penalty_area in enumerate(self.env_config["penaltyAreas"]):
            top_left = scale_coordinate(penalty_area["topLeft"])
            bottom_right = scale_coordinate(penalty_area["bottomRight"])
            length = bottom_right[0] - top_left[0]
            width = bottom_right[1] - top_left[1]
            center = [(top_left[0] + bottom_right[0]) / 2, (top_left[1] + bottom_right[1]) / 2]
            obstacle_shape=Box(length=length.item(), width=width.item())
            if penalty_area["type"]=="circle":
                radius = length/4 # The original divides by 2 (I assume this is erroneous, but I will divide it by 4 to match)
                obstacle_shape=Sphere(radius.item())
            # else:
            #     obstacle_shape=Box(length=length*2, width=width*2), # Need to multiply by two due to nature of vmas coordinate system

            print(f"Obstacle width: {width}, length: {length}, center: {center}")

            obstacle = Landmark(
                name=f"obstacle_{i}",
                collide=True,  # Penalty areas are collidable
                movable=False,
                shape=obstacle_shape, # Need to multiply by two due to nature of vmas coordinate system
                color=Color.RED,
                collision_filter=lambda e: not isinstance(e.shape, Box),
            )
            
            world.add_landmark(obstacle)
            self.obs_pos.append(torch.tensor(center, device=device))

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
        # reward = torch.zeros(
        #     self.world.batch_dim,
        #     device=self.world.device,
        #     dtype=torch.float32,
        #     )
        # Track whether the agent is currently on a waypoint
        for i, landmark in enumerate(self.world.landmarks):
            if landmark.state.pos is not None and agent.state.pos is not None:
                if landmark.name.startswith("goal"):
                    # print(i, landmark.state.pos, agent.state.pos, torch.linalg.vector_norm(landmark.state.pos - agent.state.pos), self.reward_radius)
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
            # Change angular displacement from radians to degrees
            # angular_displacement_degrees = angular_displacement * (180 / torch.pi)

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