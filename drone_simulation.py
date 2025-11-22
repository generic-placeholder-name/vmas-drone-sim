import typing
from typing import List

import json

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

def scale_coordinate(coord):
    """converts meters to [-1, 1] coordinates"""
    x, y = coord
    return torch.tensor([(x - center_x) * scale, (y - center_y) * scale])

def convert_to_original_units(scaled_coord):
    """Converts [-1, 1] coordinates to meters"""
    x, y = scaled_coord
    return [x / scale + center_x, y / scale + center_y]


def load_env_config(path):
    global center_x, center_y, scale  

    import json
    with open(path, "r") as f:
        cfg = json.load(f)

    ox1, oy1 = cfg["origBorders"]["topLeft"]
    ox2, oy2 = cfg["origBorders"]["bottomRight"]

    width  = ox2 - ox1
    height = oy2 - oy1

    center_x = (ox1 + ox2) / 2
    center_y = (oy1 + oy2) / 2
    scale    = 2 / min(width, height)

    cfg["borders"] = {
        "topLeft": scale_coordinate([ox1, oy1]),
        "bottomRight": scale_coordinate([ox2, oy2])
    }

    cols = cfg["grid"]["cols"]
    rows = cfg["grid"]["rows"]

    cell_w = width / cols
    cell_h = height / rows

    scaled_starts = []
    for col, row in cfg["startingPoints"]:
        mx = ox1 + (col - 0.5) * cell_w
        my = oy1 + (row - 0.5) * cell_h
        scaled_starts.append(scale_coordinate([mx, my]))

    cfg["startingPoints"] = scaled_starts

    return cfg

envConfig = load_env_config("envConfig.json")

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

        self.n_drones = 2
        self.drone_radius = 0.03333
        self.reward_radius = 0.01
        self.visualize_semidims = True

        # Extract world dimensions from envConfig (these are scaled coords)
        world_width = self.env_config["borders"]["bottomRight"][0]
        world_height = self.env_config["borders"]["bottomRight"][1]

        self.waypoints = []
        self.obs_pos = []
        self.drone_start_pos = []
        self.last_waypoint = {i: None for i in range(self.n_drones)}
        # Make world
        world = World(batch_dim, device, x_semidim=world_width, y_semidim=world_height)
        self._world = world
        world_dims = torch.tensor([world_width, world_height])
        self.cumulative_reward = torch.zeros(
            world.batch_dim,
            device=world.device,
            dtype=torch.float32,
        )

        self.n_boids = 3
        self.boids = []
        self.boid_start_pos = []
        self.min_distance_between_entities = 0.22
        self.x_bounds = world_width
        self.y_bounds = world_height

        # Add drones
        for i in range(self.n_drones):
            drone = Agent(
                name=f"drone_{i}",
                collide=True,
                render_action=True,
                u_range=[1, 1],
                u_multiplier=[0.05, 0.5], #[linear, angular]
                shape=Sphere(self.drone_radius),
                dynamics=DiffDrive(world, integration="rk4"),
            )
            world.add_agent(drone)
            self.drone_start_pos.append(torch.tensor(self.env_config["startingPoints"][i], device=device))

        # Add boids
        for j in range(self.n_boids):
            boid = Agent(
                name=f"boid_{j}",
                collide=True,
                render_action=False,
                u_range=[0, 0],
                u_multiplier=[0.0, 0.0], #[linear, angular]
                shape=Box(length=0.12, width=0.12),
                color=Color.BLACK,
                dynamics=DiffDrive(world, integration="rk4"),
            )
            world.add_agent(boid)
            self.boids.append(boid)
        
        self.total_rotation = torch.zeros(self.n_drones, device=device)  # Track total rotation for each drone
        self.prev_rotations = [drone.state.rot for drone in self.world.agents[: self.n_drones]]  # Track previous rotation for each drone
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
                        # if drone in point
                        world.add_landmark(goal)
                        self.waypoints.append(Waypoint(torch.tensor(convert_to_original_units(point), device=device), goal, reward_radius=self.reward_radius))
                        print(f"Waypoint {len(self.waypoints)-1} created at {point} = {convert_to_original_units(point)}")
                        break

        # Generate waypoints at start locations
        for (x, y) in self.drone_start_pos:
            point = [x.item(), y.item()]
            goal = Landmark(
                name=f"goal_{len(self.waypoints)}",
                collide=False,
                shape=Sphere(radius=self.reward_radius),
                color=Color.LIGHT_GREEN,
            )
            # if drone in point
            world.add_landmark(goal)
            self.waypoints.append(Waypoint(torch.tensor(convert_to_original_units(point), device=device), goal, reward_radius=self.reward_radius))
            print(f"Waypoint {len(self.waypoints)-1} created at {point} = {convert_to_original_units(point)}")

        self.waypoint_visits = torch.zeros([self.n_drones, len(self.waypoints)], device=device)  # Track waypoints visited by each drone
        
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

        self.prev_positions = [drone.state.pos for drone in self.world.agents[: self.n_drones]]
        self.total_distance = torch.zeros(self.n_drones, device=device)
        self.spawn_boids()
        
        return world
    
    
    def spawn_boids(self):
            # Scenario only uses a single enviornment, so we always spawn in env_index = 0
            env_index = 0
            batch_size = self.world.batch_dim
            occupied_positions_list = []

            for agent in self.world.agents:
                if agent.name.startswith("drone_"):
                    assert agent.state.pos is not None
                    assert torch.is_tensor(agent.state.pos)
                    pos = agent.state.pos.unsqueeze(1) # [batch_size, 2] -> pos shape: [batch_size, 1, 2]
                    occupied_positions_list.append(pos)
                    
            for object_pos in self.obs_pos:
                # pos shape: [2] -> pos shape: [batch_size, 2]
                pos = object_pos.to(self.world.device).expand(batch_size, -1)
                #  pos shape: [batch_size, 2] -> pos shape: [batch_size, 1, 2]
                pos = pos.unsqueeze(1)
                occupied_positions_list.append(pos)


            if occupied_positions_list:
                # get [batch_size, K, 2] where K is the number of occupied entities
                occupied_positions = torch.cat(occupied_positions_list, dim=1)
            else:
                occupied_positions = torch.Tensor()

            ScenarioUtils.spawn_entities_randomly(
                entities=self.boids,
                world=self.world,
                env_index=env_index,
                min_dist_between_entities=self.min_distance_between_entities,
                x_bounds=(-self.x_bounds, self.x_bounds),
                y_bounds=(-self.y_bounds, self.y_bounds),
                occupied_positions=occupied_positions,
            )


    def reset_world_at(self, env_index: int | None = None):
        n_goals = len(self.waypoints)
        drones = [self.world.agents[i] for i in torch.randperm(self.n_drones).tolist()]
        goals = [self.world.landmarks[i] for i in torch.range(start=0,end=n_goals-1,dtype=torch.int).tolist()]
        order = range(len(self.world.landmarks[n_goals :]))
        obstacles = [self.world.landmarks[n_goals :][i] for i in order]
        self.waypoint_visits = torch.zeros([self.n_drones, len(self.waypoints)], device=self.world.device) # reset the counter
        self.total_distance = torch.tensor([0.0 for _ in self.world.agents])
        self.total_rotation = torch.zeros(self.n_drones, device=self.world.device)  # Reset total rotation
        self.prev_rotations = [drone.state.rot for drone in self.world.agents[: self.n_drones]] # Reset previous rotations
        for i, goal in enumerate(goals):
            goal.set_pos(
                scale_coordinate(self.waypoints[i].point),
                batch_index=env_index,
            )
        for i, drone in enumerate(drones):
            drone.set_pos(
                self.drone_start_pos[i],#self.world.agents[i].state.pos,
                batch_index=env_index,
            )
        for i, obstacle in enumerate(obstacles):
            obstacle.set_pos(
                self.obs_pos[i],
                batch_index=env_index,
            )
        
        self.spawn_boids()


    def reward(self, agent: Agent):
        if not agent.name.startswith("drone_"):
            return torch.zeros(
                self.world.batch_dim,
                device=self.world.device,
                dtype=torch.float32,
            )
          
        drone_index = self.get_drone_index(agent)
        # reward = torch.zeros(
        #     self.world.batch_dim,
        #     device=self.world.device,
        #     dtype=torch.float32,
        #     )
        # Track whether the drone is currently on a waypoint
        for i, landmark in enumerate(self.world.landmarks):
            if landmark.state.pos is not None and agent.state.pos is not None:
                if landmark.name.startswith("goal"):
                    # print(i, landmark.state.pos, drone.state.pos, torch.linalg.vector_norm(landmark.state.pos - drone.state.pos), self.reward_radius)
                    if self.world.is_overlapping(agent, landmark) and self.waypoint_visits[drone_index, i] == 0:
                        waypoint_index = self.get_waypoint_index(landmark)
                        self.cumulative_reward += 1.0
                        self.waypoint_visits[drone_index, waypoint_index] += 1
                        print(f"drone {drone_index} reached waypoint {waypoint_index}!")
                        print(f"Waypoint visits: {self.waypoint_visits[drone_index]}")
                        print(f"reward: {self.cumulative_reward}")
                        print(f"total distance: {self.total_distance[drone_index]}")
                        print("----------------------------")
                elif self.world.is_overlapping(agent, landmark):
                    if landmark.collides(agent):
                        self.cumulative_reward -= self.cumulative_reward
                        print(f"Collision by drone {drone_index}")
                        print(f"reward: {self.cumulative_reward}")
                        print("----------------------------")
                        
        #Checking drone collison, with another drone.
        for i, drone2 in enumerate(self.world.agents):
            if agent != drone2 and self.world.is_overlapping(agent, drone2):
                self.cumulative_reward -= self.cumulative_reward
                print(f"drone {agent.name} collided with {drone2.name}!")
                print(f"reward: {self.cumulative_reward}")
                print("----------------------------")
        return self.cumulative_reward

    def observation(self, agent: Agent) -> torch.Tensor:
        if not agent.name.startswith("drone_"):
            # For now, just return deer position and velocity.
            pos = (agent.state.pos)
            vel = (agent.state.vel)
            return torch.cat([pos if pos is not None else torch.zeros(2, device=agent.device), vel if vel is not None else torch.zeros(2, device=agent.device)], dim=-1)
        # Update distance information
        drone_index = self.get_drone_index(agent)
        current_pos = agent.state.pos
        prev_pos = self.prev_positions[drone_index]

        # Find the distance traveled since the last step
        distance = 0.0
        if prev_pos is not None and current_pos is not None:
            distance = torch.linalg.vector_norm(current_pos - prev_pos)

        self.total_distance[drone_index] += distance
        self.prev_positions[drone_index] = current_pos

        # Update rotation information
        current_rot = agent.state.rot
        prev_rot = self.prev_rotations[drone_index]

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
            self.total_rotation[drone_index] += torch.abs(angular_displacement)


        self.prev_rotations[drone_index] = current_rot

        # Get positions of all landmarks in this drone's reference frame
        landmark_rel_poses = []
        for landmark in self.world.landmarks:
            assert landmark.state.pos is not None and agent.state.pos is not None, "Landmark or drone position is None"
            landmark_rel_poses.append(landmark.state.pos - agent.state.pos)
        return torch.cat(
            [
                agent.state.pos if agent.state.pos is not None else torch.zeros(2, device=agent.device),
                agent.state.vel if agent.state.vel is not None else torch.zeros(2, device=agent.device),
                *landmark_rel_poses,
            ],
            dim=-1,
        )
    
    def get_drone_index(self, drone: Agent):
        return int(drone.name.split("_")[1])
    
    def get_waypoint_index(self, goal: Landmark):
        return int(goal.name.split("_")[1])

    # def done(self): not implemented yet

    # def extra_render(self, env_index: int = 0):
    def extra_render(self, env_index: int = 0) -> "List[Geom]":

        geoms: List[Geom] = []

        # Agent rotation
        for drone in self.world.agents:
            geoms.append(
                ScenarioUtils.plot_entity_rotation(drone, env_index, length=0.1)
            )

        return geoms

if __name__ == "__main__":
    render_interactively(
        Scenario(), control_two_agents=True, shared_reward=True
    )
