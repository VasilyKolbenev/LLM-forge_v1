"""FSDP and DeepSpeed distributed training launcher.

Supports single-node multi-GPU and multi-node distributed training
via accelerate. Generates appropriate configs for FSDP and DeepSpeed
based on hardware and user settings.
"""

import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def launch_distributed(
    config: dict,
    script_path: Optional[str] = None,
    num_machines: int = 1,
    machine_rank: int = 0,
    master_addr: str = "127.0.0.1",
    master_port: int = 29500,
) -> dict:
    """Launch distributed training via accelerate.

    Generates accelerate config based on strategy and runs training.

    Args:
        config: Full resolved config dict.
        script_path: Path to training script. If None, uses internal launcher.
        num_machines: Number of machines in the cluster.
        machine_rank: Rank of this machine (0 = master).
        master_addr: IP address of the master node.
        master_port: Port for rendezvous.

    Returns:
        Dict with training results.
    """
    strategy = config.get("_detected_strategy", config.get("strategy"))
    num_gpus = config.get("_hardware", {}).get("num_gpus", 1)

    if strategy in ("fsdp_qlora", "fsdp_full", "fsdp_lora"):
        accel_config = _build_fsdp_config(
            config, num_machines=num_machines,
            machine_rank=machine_rank,
            master_addr=master_addr,
            master_port=master_port,
        )
    elif strategy in ("deepspeed", "deepspeed_zero2", "deepspeed_zero3"):
        accel_config = _build_deepspeed_config(
            config, num_machines=num_machines,
            machine_rank=machine_rank,
            master_addr=master_addr,
            master_port=master_port,
        )
    else:
        raise ValueError(
            f"Strategy '{strategy}' does not require distributed launcher. "
            "Use single-GPU training instead."
        )

    # Write accelerate config to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, prefix="accel_") as f:
        import yaml
        yaml.dump(accel_config, f)
        accel_config_path = f.name

    # Write training config to temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="train_config_"
    ) as f:
        import yaml
        yaml.dump(config, f)
        train_config_path = f.name

    if script_path is None:
        script_path = str(Path(__file__).parent / "_distributed_entry.py")

    cmd = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--config_file",
        accel_config_path,
        "--num_processes",
        str(num_gpus),
        "--num_machines",
        str(num_machines),
        "--machine_rank",
        str(machine_rank),
        "--main_process_ip",
        master_addr,
        "--main_process_port",
        str(master_port),
        script_path,
        "--config",
        train_config_path,
    ]

    logger.info(
        "Launching distributed training: %d GPUs × %d nodes, strategy=%s, rank=%d",
        num_gpus, num_machines, strategy, machine_rank,
    )
    logger.debug("Command: %s", " ".join(cmd))

    result = subprocess.run(cmd)

    # Clean up temp files
    Path(accel_config_path).unlink(missing_ok=True)
    Path(train_config_path).unlink(missing_ok=True)

    return {
        "status": "completed" if result.returncode == 0 else "failed",
        "exit_code": result.returncode,
        "strategy": strategy,
        "num_gpus": num_gpus,
        "num_machines": num_machines,
    }


def _build_fsdp_config(
    config: dict,
    num_machines: int = 1,
    machine_rank: int = 0,
    master_addr: str = "127.0.0.1",
    master_port: int = 29500,
) -> dict:
    """Build accelerate FSDP config with multi-node support.

    Args:
        config: Full config dict with fsdp section.
        num_machines: Number of machines.
        machine_rank: Rank of this machine.
        master_addr: Master node address.
        master_port: Master node port.

    Returns:
        Accelerate config dict.
    """
    fsdp_config = config.get("fsdp", {})
    training_config = config.get("training", {})

    sharding = fsdp_config.get("sharding_strategy", "FULL_SHARD")
    cpu_offload = fsdp_config.get("cpu_offload", False)

    accel = {
        "compute_environment": "LOCAL_MACHINE",
        "distributed_type": "FSDP",
        "fsdp_config": {
            "fsdp_sharding_strategy": sharding,
            "fsdp_offload_params": cpu_offload,
            "fsdp_auto_wrap_policy": fsdp_config.get("auto_wrap_policy", "TRANSFORMER_BASED_WRAP"),
            "fsdp_backward_prefetch_policy": fsdp_config.get("backward_prefetch", "BACKWARD_PRE"),
            "fsdp_state_dict_type": "SHARDED_STATE_DICT",
            "fsdp_sync_module_states": fsdp_config.get("sync_module_states", True),
            "fsdp_use_orig_params": True,
        },
        "mixed_precision": ("bf16" if training_config.get("bf16", True) else "no"),
        "num_machines": num_machines,
        "machine_rank": machine_rank,
        "main_process_ip": master_addr,
        "main_process_port": master_port,
        "num_processes": config.get("_hardware", {}).get("num_gpus", 2),
        "main_training_function": "main",
    }

    return accel


def _build_deepspeed_config(
    config: dict,
    num_machines: int = 1,
    machine_rank: int = 0,
    master_addr: str = "127.0.0.1",
    master_port: int = 29500,
) -> dict:
    """Build accelerate DeepSpeed config with multi-node support.

    Args:
        config: Full config dict with deepspeed section.
        num_machines: Number of machines.
        machine_rank: Rank of this machine.
        master_addr: Master node address.
        master_port: Master node port.

    Returns:
        Accelerate config dict.
    """
    ds_config = config.get("deepspeed", {})
    training_config = config.get("training", {})
    stage = ds_config.get("stage", 2)

    zero_config = {
        "zero_optimization": {
            "stage": stage,
            "offload_optimizer": {
                "device": ("cpu" if ds_config.get("cpu_offload", False) else "none"),
            },
            "offload_param": {
                "device": ("cpu" if stage == 3 and ds_config.get("cpu_offload") else "none"),
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": ds_config.get("reduce_bucket_size", 5e8),
            "stage3_prefetch_bucket_size": ds_config.get("prefetch_bucket_size", 5e8),
            "stage3_param_persistence_threshold": ds_config.get("param_persistence_threshold", 1e6),
        },
        "bf16": {"enabled": training_config.get("bf16", True)},
        "gradient_accumulation_steps": training_config.get("gradient_accumulation", 8),
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": training_config.get("batch_size", 1),
    }

    accel = {
        "compute_environment": "LOCAL_MACHINE",
        "distributed_type": "DEEPSPEED",
        "deepspeed_config": zero_config,
        "num_machines": num_machines,
        "machine_rank": machine_rank,
        "main_process_ip": master_addr,
        "main_process_port": master_port,
        "num_processes": config.get("_hardware", {}).get("num_gpus", 2),
        "main_training_function": "main",
    }

    return accel


def generate_deepspeed_config(
    stage: int = 2,
    bf16: bool = True,
    cpu_offload: bool = False,
    batch_size: int = 1,
    gradient_accumulation: int = 8,
) -> dict:
    """Generate a standalone DeepSpeed JSON config.

    Useful for manual accelerate launch or torchrun.

    Args:
        stage: ZeRO stage (2 or 3).
        bf16: Use bf16 mixed precision.
        cpu_offload: Offload optimizer/params to CPU.
        batch_size: Per-device batch size.
        gradient_accumulation: Gradient accumulation steps.

    Returns:
        DeepSpeed config dict.
    """
    config = {
        "bf16": {"enabled": bf16},
        "zero_optimization": {
            "stage": stage,
            "offload_optimizer": {
                "device": "cpu" if cpu_offload else "none",
                "pin_memory": True,
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        },
        "gradient_accumulation_steps": gradient_accumulation,
        "train_micro_batch_size_per_gpu": batch_size,
        "wall_clock_breakdown": False,
    }

    if stage == 3:
        config["zero_optimization"]["offload_param"] = {
            "device": "cpu" if cpu_offload else "none",
            "pin_memory": True,
        }
        config["zero_optimization"]["stage3_gather_16bit_weights_on_model_save"] = True

    return config


def build_multi_node_config(
    config: dict,
    num_nodes: int,
    gpus_per_node: int,
    master_addr: str,
    master_port: int = 29500,
    node_rank: int = 0,
) -> dict:
    """Build a complete distributed config for multi-node training.

    Convenience wrapper that merges distributed settings into the
    training config for use by _distributed_entry.py or RemoteJobRunner.

    Args:
        config: Base training config.
        num_nodes: Total number of machines.
        gpus_per_node: GPUs per machine.
        master_addr: Master node IP.
        master_port: Master node port.
        node_rank: This machine's rank.

    Returns:
        Config dict with distributed settings injected.
    """
    config = dict(config)
    config["_distributed"] = {
        "num_machines": num_nodes,
        "gpus_per_node": gpus_per_node,
        "master_addr": master_addr,
        "master_port": master_port,
        "node_rank": node_rank,
    }
    config.setdefault("_hardware", {})["num_gpus"] = gpus_per_node
    return config
