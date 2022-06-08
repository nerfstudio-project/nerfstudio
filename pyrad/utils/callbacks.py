# Copyright 2022 The Plenoptix Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Callback functions for training iterations
"""

from pyrad.graphs.base import Graph
from pyrad.fields.occupancy_fields.occupancy_grid import OccupancyGrid


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
