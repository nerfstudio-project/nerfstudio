"""
Callback functions for training iterations
"""

from mattport.nerf.graph.base import Graph
from mattport.nerf.occupancy_grid import OccupancyGrid


def update_occupancy(graph: Graph):
    """Update the occupancy grid depending on iteration count after every train step"""
    if not hasattr(graph, "occupancy_pointers"):
        occupancy_grids = []
        for _, module in graph.__dict__["_modules"].items():
            if isinstance(module, OccupancyGrid):
                occupancy_grids.append(module)
        graph.occupancy_pointers = occupancy_grids
    for grid in graph.occupancy_pointers:
        grid.iteration_count += 1
        if grid.iteration_count % grid.update_every_num_iters == 0:
            if grid.density_fn is not None:
                grid.update_occupancy_grid(grid.density_fn)
