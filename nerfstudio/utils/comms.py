# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

"""functionality to handle multiprocessing syncing and communicating"""

import torch.distributed as dist

LOCAL_PROCESS_GROUP = None


def is_dist_avail_and_initialized() -> bool:
    """Returns True if distributed is available and initialized."""
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    """Get total number of available gpus"""
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """Get global rank of current thread"""
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_local_rank() -> int:
    """The rank of the current process within the local (per-machine) process group."""
    if not is_dist_avail_and_initialized():
        return 0
    assert (
        LOCAL_PROCESS_GROUP is not None
    ), "Local process group is not created! Please use launch() to spawn processes!"
    return dist.get_rank(group=LOCAL_PROCESS_GROUP)


def get_local_size() -> int:
    """
    The size of the per-machine process group,
    i.e. the number of processes per machine.
    """
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size(group=LOCAL_PROCESS_GROUP)


def is_main_process() -> bool:
    """check to see if you are currently on the main process"""
    return get_rank() == 0


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if dist.get_world_size() == 1:
        return
    if dist.get_backend() == dist.Backend.NCCL:
        # This argument is needed to avoid warnings.
        # It's valid only for NCCL backend.
        dist.barrier(device_ids=[get_local_rank()])
    else:
        dist.barrier()
