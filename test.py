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
# Changed to feet x feet of farm, measured from google maps
original_top_left = [0, 0]
original_bottom_right = [800, 620]

original_width = original_bottom_right[0] - original_top_left[0]
original_height = original_bottom_right[1] - original_top_left[1]
scale_x = 1 / min(original_width, original_height)
scale_y = 1 / min(original_width, original_height)
offset_x = original_top_left[0]
offset_y = original_top_left[1]

def scale_coordinate(coord):
    x, y = coord
    scaled_x = (x - offset_x) * scale_x
    scaled_y = (y - offset_y) * scale_y
    return [scaled_x, scaled_y]

envConfig = {
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
            "topLeft": scale_coordinate([350, 620-500]),  
            "bottomRight": scale_coordinate([426, 620-400]),
            "type": "box"
        },
        {
            "topLeft": scale_coordinate([485, 620-340]),
            "bottomRight": scale_coordinate([525, 620-242]),
            "type": "box"
        },
        {
            "topLeft": scale_coordinate([350, 620-600]),
            "bottomRight": scale_coordinate([400, 620-550]),
            "type": "circle",
        },
        {
            "topLeft": scale_coordinate([175, 620-200]),
            "bottomRight": scale_coordinate([225, 620-150]),
            "type": "circle",
        }
    ],
    "startingPoints": [
        scale_coordinate([450, 200]),
        scale_coordinate([475, 200])
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
        world_dims = torch.Tensor([world_width, world_height])
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
            self.agent_start_pos.append(torch.Tensor(self.env_config["startingPoints"][i], device=device) * 2 - world_dims)
            
        self.total_rotation = torch.zeros(len(self.world.agents), device=device)  # Track total rotation for each agent
        self.prev_rotations = [agent.state.rot for agent in self.world.agents]  # Track previous rotation for each agent
      
        for (x, y) in self.agent_start_pos:
            point = torch.Tensor([x.item(), y.item()], device=device)
            goal = Landmark(
                name=f"goal {len(self.waypoints)}",
                collide=False,
                shape=Sphere(radius=self.reward_radius),
                color=Color.LIGHT_GREEN,
            )
            # if agent in point
            world.add_landmark(goal)
            self.waypoints.append(Waypoint(point, goal, reward_radius=self.reward_radius))

        # Generate goal (waypoints) points in reward areas
        print(f"World height: {world_height} \
              \nWorld width: {world_width}\n")
        
        # Generate waypoints at start locations
        for (x, y) in self.agent_start_pos:
            point = torch.Tensor([x.item(), y.item()], device=device)
            goal = Landmark(
                name=f"goal {len(self.waypoints)}",
                collide=False,
                shape=Sphere(radius=self.reward_radius),
                color=Color.LIGHT_GREEN,
            )
            # if agent in point
            world.add_landmark(goal)
            self.waypoints.append(Waypoint(point, goal, reward_radius=self.reward_radius))
        
        # Generate goal (waypoints) points in reward areas
        for x in torch.arange(self.grid_resolution/2, world_width, self.grid_resolution):
            for y in torch.arange(self.grid_resolution/2, world_height, self.grid_resolution):
                point = [x.item(), y.item()]
                print(point)
                for reward_area in self.env_config["rewardAreas"]:
                    if is_point_in_polygon(point, reward_area): # TODO: Check that point not in penalty areas
                        print("Is in reward area\n")
                        goal = Landmark(
                            name=f"goal {len(self.waypoints)}",
                            collide=False,
                            shape=Sphere(radius=self.reward_radius),
                            color=Color.LIGHT_GREEN,
                        )
                        # if agent in point
                        world.add_landmark(goal)
                        scaled_point = torch.Tensor(point, device=device) * 2 - world_dims
                        self.waypoints.append(Waypoint(scaled_point, goal, reward_radius=self.reward_radius))
                        break
        self.waypoint_visits = torch.zeros([self.n_agents, len(self.waypoints)], device=device)  # Track waypoints visited by each drone
        
        # Add penalty areas as landmarks
        for i, penalty_area in enumerate(self.env_config["penaltyAreas"]):
            top_left = penalty_area["topLeft"]
            bottom_right = penalty_area["bottomRight"]
            length = bottom_right[0] - top_left[0]
            width = bottom_right[1] - top_left[1]
            center = [(top_left[0] + bottom_right[0]) / 2, (top_left[1] + bottom_right[1]) / 2]
            obstacle_shape=Box(length=length*2, width=width*2)
            if penalty_area["type"]=="circle":
                radius = length/2
                obstacle_shape=Sphere(radius)
            # else:
            #     obstacle_shape=Box(length=length*2, width=width*2), # Need to multiply by two due to nature of vmas coordinate system


            obstacle = Landmark(
                name=f"obstacle {i}",
                collide=True,  # Penalty areas are collidable
                movable=False,
                shape=obstacle_shape, # Need to multiply by two due to nature of vmas coordinate system
                color=Color.RED,
                collision_filter=lambda e: not isinstance(e.shape, Box),
            )
            
            world.add_landmark(obstacle)
            self.obs_pos.append(torch.Tensor(center, device=device) * 2 - world_dims)

        self.prev_positions = [agent.state.pos for agent in self.world.agents]
        self.total_distance = torch.zeros(len(self.world.agents), device=device)
        
        return world
    
    def reset_world_at(self, env_index: int | None = None):
        n_goals = len(self.waypoints)
        agents = [self.world.agents[i] for i in torch.randperm(self.n_agents).tolist()]
        goals = [self.world.landmarks[i] for i in torch.randperm(n_goals).tolist()]
        order = range(len(self.world.landmarks[n_goals :]))
        obstacles = [self.world.landmarks[n_goals :][i] for i in order]
        self.waypoint_visits = torch.zeros([self.n_agents, len(self.waypoints)], device=self.world.device) # reset the counter
        self.total_distance = torch.tensor([0.0 for _ in self.world.agents])
        self.total_rotation = torch.zeros(self.n_agents, device=self.world.device)  # Reset total rotation
        self.prev_rotations = [agent.state.rot for agent in self.world.agents]  # Reset previous rotations
        for i, goal in enumerate(goals):
            goal.set_pos(
                self.waypoints[i].point,
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
                        self.cumulative_reward += 1.0
                        self.waypoint_visits[agent_index, i] += 1
                        print(f"Agent {agent_index} reached waypoint {i}!")
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