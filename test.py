import torch

from vmas import render_interactively
from vmas.simulator.core import Agent, Box, Landmark, Line, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils

# Coordinates taken from image on Google Docs.
# Reward areas are areas with crops; penalty areas are the house and the greenhouse
# Original frame coordinates
original_top_left = [181, 7]
original_bottom_right = [486, 253]

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
        "topLeft": scale_coordinate([181, 7]),
        "bottomRight": scale_coordinate([486, 253])
    },
    "rewardAreas": [
        [scale_coordinate([181, 7]), scale_coordinate([181, 253]), scale_coordinate([486, 253]), scale_coordinate([486, 7])]
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
            "topLeft": scale_coordinate([365, 100]),
            "bottomRight": scale_coordinate([378, 137])
        },
        {
            "topLeft": scale_coordinate([304, 150]),
            "bottomRight": scale_coordinate([350, 200])
        }
    ],
    "startingPoints": [
        scale_coordinate([298, 153]),
        scale_coordinate([356, 154])
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
        self.agent_u_multiplier = kwargs.pop("agent_u_multiplier", 0.5)
        ScenarioUtils.check_kwargs_consumed(kwargs)

        self.n_agents = 2
        self.agent_radius = 0.03333
        self.reward_radius = 0.01
        self.visualize_semidims = True

        # Extract world dimensions from envConfig
        world_width = self.env_config["borders"]["bottomRight"][0]
        world_height = self.env_config["borders"]["bottomRight"][1]

        self.goal_pos = []
        self.obs_pos = []
        self.agent_pos = []

        # Make world
        world = World(batch_dim, device, x_semidim=world_width, y_semidim=world_height)
        world_dims = torch.Tensor([world_width, world_height])
        self._world = world

        # Add agents
        for i in range(self.n_agents):
            agent = Agent(
                name=f"agent_{i}",
                rotatable=True,
                shape=Sphere(self.agent_radius),
                u_multiplier=self.agent_u_multiplier
            )
            world.add_agent(agent)
            self.agent_pos.append(torch.Tensor(self.env_config["startingPoints"][i], device=device) * 2 - world_dims)

        # Generate goal (waypoints) points in reward areas
        for x in torch.arange(0, world_width, self.grid_resolution):
            for y in torch.arange(0, world_height, self.grid_resolution):
                point = [x.item(), y.item()]
                for reward_area in self.env_config["rewardAreas"]:
                    if is_point_in_polygon(point, reward_area): # TODO: Check that point not in penalty areas
                        goal = Landmark(
                            name=f"goal {len(self.goal_pos)}",
                            collide=False,
                            shape=Sphere(radius=self.reward_radius),
                            color=Color.LIGHT_GREEN,
                        )
                        # if agent in point
                        world.add_landmark(goal)
                        self.goal_pos.append(torch.Tensor(point, device=device) * 2 - world_dims)
                        break
        self.waypoint_visits = torch.zeros(self.n_agents, device=device)  # Counter for waypoint visits

        self.prev_positions = {agent.name: agent.state.pos.clone() for agent in self.world.agents}  # Store initial positions
        self.total_distance = {agent.name: 0.0 for agent in self.world.agents}  # Track total distance
        # Add penalty areas as landmarks
        for i, penalty_area in enumerate(self.env_config["penaltyAreas"]):
            top_left = penalty_area["topLeft"]
            bottom_right = penalty_area["bottomRight"]
            length = bottom_right[0] - top_left[0]
            width = bottom_right[1] - top_left[1]
            center = [(top_left[0] + bottom_right[0]) / 2, (top_left[1] + bottom_right[1]) / 2]

            obs = Landmark(
                name=f"obstacle {i}",
                collide=True,  # Penalty areas are collidable
                movable=False,
                shape=Box(length=length, width=width),
                color=Color.RED,
                collision_filter=lambda e: not isinstance(e.shape, Box),
            )
            
            world.add_landmark(obs)
            self.obs_pos.append(torch.Tensor(center, device=device) * 2 - world_dims)

        return world

    def reset_world_at(self, env_index: int | None = None):
        n_goals = len(self.goal_pos)
        agents = [self.world.agents[i] for i in torch.randperm(self.n_agents).tolist()]
        goals = [self.world.landmarks[i] for i in torch.randperm(n_goals).tolist()]
        order = range(len(self.world.landmarks[n_goals :]))
        obstacles = [self.world.landmarks[n_goals :][i] for i in order]
        self.waypoint_visits = torch.zeros(self.n_agents, device=self.world.device) # reset the counter
        for i, goal in enumerate(goals):
            goal.set_pos(
                self.goal_pos[i],
                batch_index=env_index,
            )
        for i, agent in enumerate(agents):
            agent.set_pos(
                self.agent_pos[i],
                batch_index=env_index,
            )
        for i, obstacle in enumerate(obstacles):
            obstacle.set_pos(
                self.obs_pos[i],
                batch_index=env_index,
            )

    def reward(self, agent: Agent): # dummy function, which does nothing for now
        rew = torch.zeros(
            self.world.batch_dim,
            device=self.world.device,
            dtype=torch.float32,
            )
        for i, goal in enumerate(self.goal_pos):
            if torch.norm(agent.state.pos - goal) < self.reward_radius:
                rew += 1.0 
                agent_index = int(agent.name.split("_")[1])
                self.waypoint_visits[agent_index] += 1  # Increment counter
        for agent in self.world.agents:
            agent_index = int(agent.name.split("_")[1])
            # print(f"Agent {agent.name} waypoint visits: {self.waypoint_visits[agent_index].item()}")
            print(f"Agent {agent.name} total distance: {self.total_distance[agent.name]:.2f}")
        return rew

    def observation(self, agent: Agent):
        # Get the current position
        current_pos = agent.state.pos

        # Get the previous position
        prev_pos = self.prev_positions[agent.name]

        # Find the distance traveled since the last step
        distance = torch.norm(current_pos - prev_pos).item()  # Euclidean distance

        # Update the total distance covered
        self.total_distance[agent.name] += distance

        # Update the previous spot to the current spot for the next step
        self.prev_positions[agent.name] = current_pos.clone()

        # get positions of all entities in this agent's reference frame
        landmarks = self.world.landmarks[self.n_agents :]
        return torch.cat(
            [
                agent.state.pos,
                agent.state.vel,
                *[landmark.state.pos - agent.state.pos for landmark in landmarks],
            ],
            dim=-1,
        )

    # def done(self): not implemented yet

    # def extra_render(self, env_index: int = 0):


if __name__ == "__main__":
    render_interactively(
        Scenario(), control_two_agents=True, shared_reward=False
    )