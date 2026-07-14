# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Post-initialization memory report utility.

This module provides a function to collect, aggregate, and log a detailed
memory consumption report after the model is fully loaded and all GPU
resources (weights, KV cache, CUDA context, CUDA graphs, etc.) are allocated.

The report is emitted **once**, immediately before the HTTP API endpoints
are listed by :func:`vllm.entrypoints.launcher.serve_http`.

Design goals
------------
* **Production-ready**: every external call is wrapped in try/except so a
  failure in one metric never prevents the rest of the report (or server
  startup) from proceeding.
* **Minimally invasive**: the only integration point is a single call to
  :func:`log_memory_report` inserted in ``serve_http``.
* **Multi-GPU / TP / DP aware**: per-worker GPU data is gathered via
  ``collective_rpc``; results are aggregated and a summary line is printed.
* **Graceful NVML absence**: falls back to PyTorch CUDA APIs when NVML is
  unavailable (e.g. Jetson, some WSL configs).
"""

from __future__ import annotations

import os
from typing import Any

import psutil
import torch

from vllm.logger import init_logger
from vllm.utils.mem_constants import GiB_bytes, MiB_bytes

logger = init_logger(__name__)


def _gib(b: int | float | None) -> str:
    """Format bytes as GiB string, handling None."""
    if b is None:
        return "N/A"
    return f"{round(b / GiB_bytes, 2)} GiB"


def _mib(b: int | float | None) -> str:
    """Format bytes as MiB string, handling None."""
    if b is None:
        return "N/A"
    return f"{round(b / MiB_bytes, 2)} MiB"


# ---------------------------------------------------------------------------
# Worker-side collection (runs inside each worker process)
# ---------------------------------------------------------------------------

def collect_worker_memory_report() -> dict[str, Any]:
    """Collect per-GPU memory data from within a worker process.

    This function is designed to be called via ``collective_rpc`` so it
    executes in the worker process that owns the CUDA context.

    Returns a dict with keys:
        - ``rank``: worker rank
        - ``local_rank``: local device rank
        - ``device``: device index (int)
        - ``gpu_name``: GPU name (str or None)
        - ``gpu_uuid``: GPU UUID (str or None)
        - ``vram_total``: total VRAM in bytes (int or None)
        - ``vram_used``: used VRAM in bytes (int or None)
        - ``vram_free``: free VRAM in bytes (int or None)
        - ``nvml_available``: whether NVML was used (bool)
        - ``torch_allocated``: PyTorch allocator allocated bytes
        - ``torch_reserved``: PyTorch allocator reserved bytes
        - ``torch_peak_allocated``: PyTorch peak allocated bytes
        - ``torch_peak_reserved``: PyTorch peak reserved bytes
        - ``weights_memory``: model weights memory in bytes (or None)
        - ``kv_cache_memory``: KV cache budget in bytes (or None)
        - ``peak_activation_memory``: peak activation memory (or None)
        - ``non_torch_memory``: non-torch GPU memory (or None)
        - ``cudagraph_memory_estimate``: CUDA graph memory estimate (or None)
        - ``init_free_memory``: initial free memory snapshot (or None)
        - ``init_total_memory``: initial total memory snapshot (or None)
        - ``requested_memory``: requested GPU memory budget (or None)
        - ``error``: error message if collection failed (str or None)
    """
    result: dict[str, Any] = {
        "error": None,
    }

    try:
        from vllm.distributed.parallel_state import get_pp_group, get_tp_group

        try:
            rank = get_tp_group().rank
        except Exception:
            rank = -1
        try:
            local_rank = int(os.environ.get("LOCAL_RANK", -1))
        except Exception:
            local_rank = -1
        result["rank"] = rank
        result["local_rank"] = local_rank
    except Exception:
        result["rank"] = -1
        result["local_rank"] = -1

    # --- NVML data (total/used/free VRAM) ---
    nvml_available = False
    try:
        from vllm.platforms import current_platform

        if current_platform.is_cuda():
            from vllm.utils.import_utils import import_pynvml

            pynvml = import_pynvml()
            pynvml.nvmlInit()
            try:
                # Get the physical device index for this worker
                device_index = torch.cuda.current_device()
                physical_id = (
                    current_platform.device_id_to_physical_device_id(device_index)
                )
                handle = pynvml.nvmlDeviceGetHandleByIndex(physical_id)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                result["device"] = device_index
                gpu_name = pynvml.nvmlDeviceGetName(handle)
                if not isinstance(gpu_name, str):
                    gpu_name = gpu_name.decode("utf-8", "replace")
                result["gpu_name"] = gpu_name
                gpu_uuid = pynvml.nvmlDeviceGetUUID(handle)
                if not isinstance(gpu_uuid, str):
                    gpu_uuid = gpu_uuid.decode("utf-8", "replace")
                result["gpu_uuid"] = gpu_uuid
                result["vram_total"] = int(mem_info.total)
                result["vram_used"] = int(mem_info.used)
                result["vram_free"] = int(mem_info.free)
                result["nvml_available"] = True
                nvml_available = True
            finally:
                pynvml.nvmlShutdown()
    except Exception:
        pass

    # --- Fallback: use torch.cuda for VRAM info if NVML failed ---
    if not nvml_available:
        try:
            if torch.cuda.is_available():
                device_index = torch.cuda.current_device()
                props = torch.cuda.get_device_properties(device_index)
                free, total = torch.cuda.mem_get_info(device_index)
                result["device"] = device_index
                result["gpu_name"] = props.name
                result["gpu_uuid"] = None
                result["vram_total"] = int(total)
                result["vram_used"] = int(total - free)
                result["vram_free"] = int(free)
                result["nvml_available"] = False
        except Exception:
            pass

    # --- PyTorch CUDA allocator stats ---
    try:
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            result["torch_allocated"] = torch.cuda.memory_allocated(device)
            result["torch_reserved"] = torch.cuda.memory_reserved(device)
            result["torch_peak_allocated"] = torch.cuda.max_memory_allocated(device)
            result["torch_peak_reserved"] = torch.cuda.max_memory_reserved(device)
        else:
            result["torch_allocated"] = 0
            result["torch_reserved"] = 0
            result["torch_peak_allocated"] = 0
            result["torch_peak_reserved"] = 0
    except Exception:
        result["torch_allocated"] = None
        result["torch_reserved"] = None
        result["torch_peak_allocated"] = None
        result["torch_peak_reserved"] = None

    # --- vLLM internal memory metrics (from Worker attributes) ---
    # These are set on the Worker instance. Since this function is called
    # via collective_rpc on the worker, we access them through the worker
    # object. However, collective_rpc calls a method by name on the worker,
    # so this function should be a method of Worker. We handle the case
    # where it's called standalone (returning None for internal metrics).
    result["weights_memory"] = None
    result["kv_cache_memory"] = None
    result["peak_activation_memory"] = None
    result["non_torch_memory"] = None
    result["cuda_graph_memory"] = None
    result["cudagraph_memory_estimate"] = None
    result["workspace_memory"] = None
    result["custom_allreduce_memory"] = None
    result["nccl_symmetric_memory"] = None
    result["non_torch_unattributed"] = None
    result["init_free_memory"] = None
    result["init_total_memory"] = None
    result["requested_memory"] = None

    return result


# ---------------------------------------------------------------------------
# System memory collection (runs in the API server process)
# ---------------------------------------------------------------------------

def _collect_system_memory() -> dict[str, Any]:
    """Collect system (host RAM) memory metrics for the current process."""
    result: dict[str, Any] = {}

    # --- System-wide RAM ---
    try:
        vm = psutil.virtual_memory()
        result["ram_total"] = vm.total
        result["ram_used"] = vm.used
        result["ram_available"] = vm.available
    except Exception:
        result["ram_total"] = None
        result["ram_used"] = None
        result["ram_available"] = None

    # --- Process-level memory ---
    try:
        proc = psutil.Process()
        mem_info = proc.memory_info()
        result["rss"] = mem_info.rss
        result["vms"] = mem_info.vms
        # shared memory from psutil (platform-dependent)
        result["shared_memory"] = getattr(mem_info, "shared", None)
        # private memory = RSS - shared
        shared = getattr(mem_info, "shared", 0) or 0
        result["private_memory"] = mem_info.rss - shared
    except Exception:
        result["rss"] = None
        result["vms"] = None
        result["shared_memory"] = None
        result["private_memory"] = None

    # --- Pinned host memory (PyTorch CUDA pinned memory) ---
    # PyTorch does not expose a direct API for tracking pinned host memory.
    # We attempt to read it from the CUDA host allocator stats if available.
    try:
        if torch.cuda.is_available():
            # The CUDA caching host allocator exposes stats via
            # torch.cuda.memory_stats() with the "host" prefix on some
            # PyTorch versions. If not available, we report 0.
            host_stats = torch.cuda.memory_stats()
            pinned = host_stats.get("host.all.allocated", 0)
            if pinned == 0:
                # Fallback: try the alternative key naming
                pinned = host_stats.get("allocated_bytes.host.all.current", 0)
            result["pinned_host_memory"] = pinned
        else:
            result["pinned_host_memory"] = 0
    except Exception:
        result["pinned_host_memory"] = None

    return result


# ---------------------------------------------------------------------------
# Aggregation and formatting
# ---------------------------------------------------------------------------

def _format_gpu_block(worker_report: dict[str, Any]) -> list[str]:
    """Format a single GPU's memory data as a list of log lines."""
    lines: list[str] = []
    rank = worker_report.get("rank", -1)
    local_rank = worker_report.get("local_rank", -1)
    device = worker_report.get("device", "?")
    gpu_name = worker_report.get("gpu_name", "Unknown")
    gpu_uuid = worker_report.get("gpu_uuid", "N/A")
    nvml = worker_report.get("nvml_available", False)

    header = f"  GPU {device}"
    if rank >= 0:
        header += f" (rank={rank}"
    if local_rank >= 0:
        header += f", local_rank={local_rank}"
    if rank >= 0:
        header += ")"
    header += f": {gpu_name}"
    if gpu_uuid and gpu_uuid != "N/A":
        header += f" [UUID: {gpu_uuid}]"
    if not nvml:
        header += " (NVML unavailable — using PyTorch fallback)"

    lines.append(header)
    lines.append(f"    {'Metric':<35} {'Value':<20}")
    lines.append(f"    {'─' * 55}")

    # VRAM (NVML or fallback)
    vram_total = worker_report.get("vram_total")
    vram_used = worker_report.get("vram_used")
    vram_free = worker_report.get("vram_free")
    lines.append(f"    {'Total VRAM':<35} {_gib(vram_total):<20}")
    lines.append(f"    {'Used VRAM':<35} {_gib(vram_used):<20}")
    lines.append(f"    {'Free VRAM':<35} {_gib(vram_free):<20}")

    # PyTorch CUDA allocator
    lines.append(f"    {'─' * 55}")
    lines.append(f"    {'CUDA Allocator (Allocated)':<35} {_gib(worker_report.get('torch_allocated')):<20}")
    lines.append(f"    {'CUDA Allocator (Reserved)':<35} {_gib(worker_report.get('torch_reserved')):<20}")
    lines.append(f"    {'CUDA Allocator (Peak Allocated)':<35} {_gib(worker_report.get('torch_peak_allocated')):<20}")
    lines.append(f"    {'CUDA Allocator (Peak Reserved)':<35} {_gib(worker_report.get('torch_peak_reserved')):<20}")

    # vLLM internal metrics
    lines.append(f"    {'─' * 55}")
    weights = worker_report.get("weights_memory")
    kv_cache = worker_report.get("kv_cache_memory")
    peak_act = worker_report.get("peak_activation_memory")
    non_torch = worker_report.get("non_torch_memory")
    cuda_graph = worker_report.get("cuda_graph_memory")
    cudagraph_est = worker_report.get("cudagraph_memory_estimate")
    workspace = worker_report.get("workspace_memory")
    custom_ar = worker_report.get("custom_allreduce_memory")
    nccl_symm = worker_report.get("nccl_symmetric_memory")
    non_torch_unattr = worker_report.get("non_torch_unattributed")
    init_free = worker_report.get("init_free_memory")
    init_total = worker_report.get("init_total_memory")
    requested = worker_report.get("requested_memory")

    lines.append(f"    {'Model Weights':<35} {_gib(weights):<20}")
    lines.append(f"    {'KV Cache (budget)':<35} {_gib(kv_cache):<20}")
    lines.append(f"    {'Peak Activation Memory':<35} {_gib(peak_act):<20}")
    lines.append(f"    {'CUDA Graph Memory (actual)':<35} {_gib(cuda_graph):<20}")
    lines.append(f"    {'CUDA Graph Memory (pre-estimate)':<35} {_gib(cudagraph_est):<20}")

    # Non-torch memory breakdown
    lines.append(f"    {'─' * 55}")
    lines.append(f"    {'Non-Torch GPU Memory (total)':<35} {_gib(non_torch):<20}")
    lines.append(f"    {'  Workspace':<35} {_gib(workspace):<20}")
    lines.append(f"    {'  Custom All-Reduce Buffers':<35} {_gib(custom_ar):<20}")
    lines.append(f"    {'  NCCL Symmetric Memory Pool':<35} {_gib(nccl_symm):<20}")
    lines.append(f"    {'  Unattributed (CUDA Ctx, NCCL, etc.)':<35} {_gib(non_torch_unattr):<20}")

    lines.append(f"    {'─' * 55}")
    lines.append(f"    {'Init Free Memory (snapshot)':<35} {_gib(init_free):<20}")
    lines.append(f"    {'Init Total Memory (snapshot)':<35} {_gib(init_total):<20}")
    lines.append(f"    {'Requested Memory (gpu_mem_util)':<35} {_gib(requested):<20}")

    return lines


def _format_system_block(sys_mem: dict[str, Any]) -> list[str]:
    """Format system memory data as a list of log lines."""
    lines: list[str] = []
    lines.append("System Memory (RAM)")
    lines.append(f"  {'Metric':<35} {'Value':<20}")
    lines.append(f"  {'─' * 55}")

    lines.append(f"  {'Total RAM':<35} {_gib(sys_mem.get('ram_total')):<20}")
    lines.append(f"  {'Used RAM':<35} {_gib(sys_mem.get('ram_used')):<20}")
    lines.append(f"  {'Available RAM':<35} {_gib(sys_mem.get('ram_available')):<20}")

    lines.append(f"  {'─' * 55}")
    lines.append(f"  {'Process RSS':<35} {_gib(sys_mem.get('rss')):<20}")
    lines.append(f"  {'Process Virtual Memory (VMS)':<35} {_gib(sys_mem.get('vms')):<20}")
    lines.append(f"  {'Process Shared Memory':<35} {_gib(sys_mem.get('shared_memory')):<20}")
    lines.append(f"  {'Process Private Memory':<35} {_gib(sys_mem.get('private_memory')):<20}")

    pinned = sys_mem.get("pinned_host_memory")
    if pinned is not None and pinned > 0:
        lines.append(f"  {'Pinned Host Memory':<35} {_mib(pinned):<20}")
    else:
        lines.append(f"  {'Pinned Host Memory':<35} {'0 MiB (not in use)':<20}")

    return lines


def _format_summary(worker_reports: list[dict[str, Any]]) -> list[str]:
    """Format the aggregate summary across all GPUs."""
    lines: list[str] = []
    total_vram = 0
    used_vram = 0
    free_vram = 0
    total_weights = 0
    total_kv_cache = 0
    total_torch_allocated = 0
    total_torch_reserved = 0
    count = 0

    total_cuda_graph = 0
    total_non_torch = 0
    for wr in worker_reports:
        if wr.get("vram_total") is not None:
            total_vram += wr["vram_total"]
            used_vram += wr.get("vram_used", 0) or 0
            free_vram += wr.get("vram_free", 0) or 0
            count += 1
        if wr.get("weights_memory") is not None:
            total_weights += wr["weights_memory"]
        if wr.get("kv_cache_memory") is not None:
            total_kv_cache += wr["kv_cache_memory"]
        if wr.get("torch_allocated") is not None:
            total_torch_allocated += wr["torch_allocated"]
        if wr.get("torch_reserved") is not None:
            total_torch_reserved += wr["torch_reserved"]
        if wr.get("cuda_graph_memory") is not None:
            total_cuda_graph += wr["cuda_graph_memory"]
        if wr.get("non_torch_memory") is not None:
            total_non_torch += wr["non_torch_memory"]

    lines.append(f"  GPUs reported: {count}")
    lines.append(f"  {'Total VRAM (all GPUs)':<35} {_gib(total_vram):<20}")
    lines.append(f"  {'Used VRAM (all GPUs)':<35} {_gib(used_vram):<20}")
    lines.append(f"  {'Free VRAM (all GPUs)':<35} {_gib(free_vram):<20}")
    lines.append(f"  {'Model Weights (all GPUs)':<35} {_gib(total_weights):<20}")
    lines.append(f"  {'KV Cache budget (all GPUs)':<35} {_gib(total_kv_cache):<20}")
    lines.append(f"  {'CUDA Graph Memory (all GPUs)':<35} {_gib(total_cuda_graph):<20}")
    lines.append(f"  {'Non-Torch GPU Memory (all GPUs)':<35} {_gib(total_non_torch):<20}")
    lines.append(f"  {'CUDA Alloc. Allocated (all GPUs)':<35} {_gib(total_torch_allocated):<20}")
    lines.append(f"  {'CUDA Alloc. Reserved (all GPUs)':<35} {_gib(total_torch_reserved):<20}")

    return lines


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def log_memory_report(engine_client: Any) -> None:
    """Collect and log a detailed memory consumption report.

    This function queries all worker processes via ``collective_rpc`` to
    gather per-GPU memory data, collects system memory metrics locally,
    and logs a single consolidated report.

    Args:
        engine_client: An :class:`~vllm.engine.protocol.EngineClient` instance
            (e.g. :class:`~vllm.v1.engine.async_llm.AsyncLLM`) that supports
            ``collective_rpc``.
    """
    separator = "=" * 70

    logger.info(separator)
    logger.info("MEMORY CONSUMPTION REPORT (post model initialization)")
    logger.info(separator)

    # --- Collect per-worker GPU memory data ---
    worker_reports: list[dict[str, Any]] = []
    try:
        results = await engine_client.collective_rpc("get_memory_report")
        if results:
            for r in results:
                if isinstance(r, dict):
                    worker_reports.append(r)
                elif isinstance(r, Exception):
                    logger.warning(
                        "Failed to collect memory report from a worker: %s", r
                    )
                else:
                    logger.warning(
                        "Unexpected memory report type from worker: %s", type(r)
                    )
    except NotImplementedError:
        logger.info(
            "  (collective_rpc not supported by this engine client — "
            "skipping per-GPU worker memory data)"
        )
    except Exception as e:
        logger.warning("Failed to collect per-worker memory reports: %s", e)

    # --- Log per-GPU blocks ---
    if worker_reports:
        # Sort by rank for deterministic output
        worker_reports.sort(key=lambda r: r.get("rank", -1))

        for wr in worker_reports:
            if wr.get("error"):
                logger.warning(
                    "Worker rank=%s memory report error: %s",
                    wr.get("rank", "?"),
                    wr["error"],
                )
                continue
            for line in _format_gpu_block(wr):
                logger.info(line)
            logger.info("")

        # --- Summary across all GPUs ---
        logger.info("GPU Memory Summary (all GPUs)")
        logger.info(f"  {'─' * 55}")
        for line in _format_summary(worker_reports):
            logger.info(line)
    else:
        logger.info("  (No worker memory data available)")
        logger.info(
            "  Reason: collective_rpc did not return any worker reports. "
            "This can happen with InprocClient or if the engine client "
            "does not support collective_rpc."
        )

    logger.info("")
    logger.info(separator)

    # --- System memory ---
    try:
        sys_mem = _collect_system_memory()
        for line in _format_system_block(sys_mem):
            logger.info(line)
    except Exception as e:
        logger.warning("Failed to collect system memory data: %s", e)

    logger.info(separator)
