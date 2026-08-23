# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os

import ttnn
from models.common.utility_functions import is_blackhole

CCL_NUM_LINKS_ENV = "GEMMA4_CCL_NUM_LINKS"
CCL_TOPOLOGY_ENV = "GEMMA4_CCL_TOPOLOGY"
CCL_ASYNC_ENV = "GEMMA4_CCL_ASYNC"


def default_num_links():
    """Default TP-collective link count for the current arch.

    Blackhole boards expose 2 ethernet links between adjacent mesh devices, so
    reduce-scatter / all-gather can run at ~2x bandwidth vs a single link — and
    on Gemma4 prefill the per-layer all-reduces are ~31% of device time, so this
    is the single highest-ROI CCL knob. Wormhole (T3K/Galaxy) defaults to 1 link
    here (its multi-link tuning needs a separate sweep). An explicit
    ``GEMMA4_CCL_NUM_LINKS`` overrides the arch default.
    """
    env = os.environ.get(CCL_NUM_LINKS_ENV)
    if env is not None:
        value = int(str(env).strip())
        if value < 1:
            raise ValueError(f"{CCL_NUM_LINKS_ENV} must be a positive integer, got {value!r}")
        return value
    return 2 if is_blackhole() else 1


def _resolve_topology(value=None):
    """Resolve the CCL fabric topology (``GEMMA4_CCL_TOPOLOGY``). Default Linear."""
    if value is None:
        value = os.environ.get(CCL_TOPOLOGY_ENV, "linear")
    if isinstance(value, ttnn.Topology):
        return value
    normalized = str(value).strip().lower()
    if normalized in ("linear", "0", "false"):
        return ttnn.Topology.Linear
    if normalized in ("ring", "1", "true"):
        return ttnn.Topology.Ring
    raise ValueError(f"{CCL_TOPOLOGY_ENV} must be 'linear' or 'ring', got {value!r}")


def _resolve_async(value=None):
    """Resolve the persistent-async CCL flag (``GEMMA4_CCL_ASYNC``). Default off."""
    if value is None:
        value = os.environ.get(CCL_ASYNC_ENV, "0")
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in ("0", "false", "no", "off"):
        return False
    if normalized in ("1", "true", "yes", "on"):
        return True
    raise ValueError(f"{CCL_ASYNC_ENV} must be a boolean, got {value!r}")


class CCLManager:
    """CCL manager for Gemma4 tensor parallelism.

    Stores mesh_device reference and num_links for CCL operations.
    Semaphores are retained for the experimental async CCL path (see TODO below).
    """

    def __init__(self, mesh_device, num_links=None, topology=None):
        if num_links is None:
            num_links = default_num_links()
        self.mesh_device = mesh_device
        self.num_links = num_links
        self.topology = _resolve_topology(topology)
        self.async_ = _resolve_async()
        self.num_devices = mesh_device.get_num_devices()

        # Semaphores for the persistent async CCL path (reduce_scatter +
        # all_gather with preallocated double-buffered semaphores). The simple
        # ttnn.all_reduce / ttnn.all_gather path is the functional default;
        # GEMMA4_CCL_ASYNC=1 enables the persistent async ops.
        grid = mesh_device.compute_with_storage_grid_size()
        num_cores = grid.x * grid.y
        core_range_set = ttnn.num_cores_to_corerangeset(num_cores, grid, row_wise=True)

        self._rs_semaphores = []
        self._ag_semaphores = []
        self._barrier_semaphores = []
        for _ in range(2):
            self._rs_semaphores.append([ttnn.create_global_semaphore(mesh_device, core_range_set, 0) for _ in range(3)])
            self._ag_semaphores.append([ttnn.create_global_semaphore(mesh_device, core_range_set, 0) for _ in range(2)])
            self._barrier_semaphores.append(ttnn.create_global_semaphore(mesh_device, core_range_set, 0))
        ttnn.synchronize_device(mesh_device)

        self._rs_idx = 0
        self._ag_idx = 0
        self._barrier_idx = 0

    def get_rs_semaphore(self):
        """Returns list of 3 semaphores for reduce_scatter (cycles double-buffer)."""
        sems = self._rs_semaphores[self._rs_idx]
        self._rs_idx = (self._rs_idx + 1) % 2
        return sems

    def get_ag_semaphore(self):
        """Returns list of 2 semaphores for all_gather (cycles double-buffer)."""
        sems = self._ag_semaphores[self._ag_idx]
        self._ag_idx = (self._ag_idx + 1) % 2
        return sems

    def get_barrier_semaphore(self):
        """Returns single barrier semaphore (cycles double-buffer)."""
        sem = self._barrier_semaphores[self._barrier_idx]
        self._barrier_idx = (self._barrier_idx + 1) % 2
        return sem


def ccl_allreduce(tensor, mesh_config, ccl_manager, memory_config=None):
    """All-reduce across TP devices."""
    if mesh_config is None or mesh_config.tp <= 1:
        return tensor

    memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG
    tp_axis = mesh_config.tp_axis

    if ccl_manager.async_:
        # Persistent async all-reduce: reduce-scatter + all-gather with
        # preallocated double-buffered semaphores so the initial global
        # ownership synchronization is skipped.
        scattered = ttnn.experimental.reduce_scatter_minimal_async(
            tensor,
            dim=3,
            cluster_axis=tp_axis,
            num_links=ccl_manager.num_links,
            topology=ccl_manager.topology,
            multi_device_global_semaphore=ccl_manager.get_rs_semaphore(),
            barrier_semaphore=ccl_manager.get_barrier_semaphore(),
            memory_config=memory_config,
        )
        tensor.deallocate(True)
        gathered = ttnn.experimental.all_gather_async(
            scattered,
            dim=3,
            cluster_axis=tp_axis,
            mesh_device=ccl_manager.mesh_device,
            num_links=ccl_manager.num_links,
            topology=ccl_manager.topology,
            multi_device_global_semaphore=ccl_manager.get_ag_semaphore(),
            barrier_semaphore=ccl_manager.get_barrier_semaphore(),
            memory_config=memory_config,
        )
        scattered.deallocate(True)
        return gathered

    result = ttnn.all_reduce(
        tensor,
        cluster_axis=tp_axis,
        num_links=ccl_manager.num_links,
        topology=ccl_manager.topology,
        memory_config=memory_config,
    )
    tensor.deallocate(True)
    return result


def ccl_allgather(tensor, mesh_config, ccl_manager, dim=3, memory_config=None):
    """All-gather across TP devices."""
    if mesh_config is None or mesh_config.tp <= 1:
        return tensor

    memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG
    tp_axis = mesh_config.tp_axis

    if ccl_manager.async_:
        gathered = ttnn.experimental.all_gather_async(
            tensor,
            dim=dim,
            cluster_axis=tp_axis,
            mesh_device=ccl_manager.mesh_device,
            num_links=ccl_manager.num_links,
            topology=ccl_manager.topology,
            multi_device_global_semaphore=ccl_manager.get_ag_semaphore(),
            barrier_semaphore=ccl_manager.get_barrier_semaphore(),
            memory_config=memory_config,
        )
        tensor.deallocate(True)
        return gathered

    gathered = ttnn.all_gather(
        tensor,
        dim=dim,
        cluster_axis=tp_axis,
        num_links=ccl_manager.num_links,
        topology=ccl_manager.topology,
        memory_config=memory_config,
    )
    tensor.deallocate(True)
    return gathered
