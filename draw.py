from graph import Waypoint, Edge, Elbow
from vmas.simulator.core import Agent, Box, Landmark, Line, Sphere, World

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch

def draw_paths(
    waypoints: list[Waypoint],
    paths: list[list[Edge]],
    obstacles: None | list[Landmark],
    dimensions: tuple[float, float, float, float],
    file_path: str,
):
    """
    Draws specified paths on a blank image and saves it to the specified file path.
    :param waypoints: List of Waypoint objects.
    :param paths: List of paths, each path is a list of Edge objects.
    :param obstacles: List of Landmark objects representing obstacles.
                      - Box instances have x, y, width, height.
                      - Sphere instances have x, y, radius.
    :param dimensions: A tuple (x_min, y_min, x_max, y_max) defining the bounding box.
    :param file_path: File path to save the image.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw free area
    x_min, y_min, x_max, y_max = dimensions
    free_width = x_max - x_min
    free_height = y_max - y_min
    free_area = patches.Rectangle(
        (x_min, y_min), free_width, free_height,
        facecolor='green', alpha=0.2, edgecolor='none'
    )
    ax.add_patch(free_area)

    # Draw obstacles by type
    if obstacles:
        for obs in obstacles:
            if isinstance(obs, Box): 
                # draw rectangle centered at (x, y)
                # TODO: AFAIK, Landmark objects do not have (x, y) attributes. Explore potential solutions.
                ll = (obs.x - obs.width/2, obs.y - obs.height/2)
                rect = patches.Rectangle(
                    ll, obs.width, obs.height,
                    facecolor='gray', alpha=0.5, edgecolor='gray'
                )
                ax.add_patch(rect)

            elif isinstance(obs, Sphere):
                circle = patches.Circle(
                    (obs.x, obs.y), obs.radius,
                    facecolor='gray', alpha=0.5, edgecolor='gray'
                )
                ax.add_patch(circle)

            else:
                # fallback: draw as a point
                ax.scatter(obs.x, obs.y, c='gray', s=30, alpha=0.5)

    # Plot waypoints
    if waypoints:
        coords = torch.stack([wp.point for wp in waypoints]).numpy()
        ax.scatter(coords[:, 0], coords[:, 1],
                   c='black', s=10, alpha=0.5, label='Waypoints')

    # Plot paths
    styles = ['r-', 'b-', 'm-', 'c-']
    for i, path in enumerate(paths):
        if not path:
            continue
        xs, ys = [], []
        current = path[0].node1
        xs.append(current.point[0].item())
        ys.append(current.point[1].item())
        for edge in path:
            nxt = edge.node2 if edge.node1 == current else edge.node1
            xs.append(nxt.point[0].item())
            ys.append(nxt.point[1].item())
            current = nxt
        ax.plot(xs, ys, styles[i % len(styles)],
                lw=2, label=f'Drone {i+1} Path')

    # Compute & display metrics
    def path_length(p):
        return sum(e.length.item() for e in p)

    def path_turns(p):
        tot = 0
        for i in range(1, len(p)):
            nxt = (i + 1) % len(p)
            elbow = Elbow(p[i], p[nxt])
            tot += elbow.angle()
        return tot

    stats = []
    for idx, path in enumerate(paths):
        if path:
            L = path_length(path)
            T = path_turns(path)
            stats.append(f"Drone {idx+1}: Length={L:.1f}, Turns={T:.1f}°")

    if stats:
        ax.text(0.05, 0.95, "\n".join(stats),
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.8))

    # 6) Final styling
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=len(paths) or 1)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.set_title("Survey Path")

    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
