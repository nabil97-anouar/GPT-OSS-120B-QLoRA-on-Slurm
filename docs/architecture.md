# Architecture

## Overview

The project separates scheduler orchestration, distributed launch, model
training, and artifact export into small components with clear ownership.

```text
sbatch
  `-- run_sft_24gpus_gptoss.slurm
        `-- srun: one task per node
              `-- launch_torchrun.sh
                    `-- torchrun: one worker per GPU
                          `-- train_lora_gptoss_unsloth.py
```

## Process topology

The default Slurm job requests three nodes and eight H100 GPUs per node. `srun`
places one launcher on each node, and each launcher starts eight Python workers.
This produces a 24-process NCCL group.

The first allocated host provides the rendezvous address. `torchrun` supplies
`RANK`, `LOCAL_RANK`, and `WORLD_SIZE` to each worker. The Python entry point
uses those values directly, selects the matching CUDA device, and verifies the
process group before model initialization.

## Training path

Each worker performs the same sequence:

1. Initialize the NCCL process group.
2. Load the 4-bit GPT-OSS checkpoint on its local device.
3. Attach LoRA adapters to attention and MLP projection layers.
4. Normalize JSONL records into chat messages.
5. Render the model chat template and pack sequences.
6. Train with synchronized gradients through PyTorch Distributed.

The global batch size is the product of world size, per-device batch size, and
gradient accumulation steps.

## Artifact ownership

Training checkpoints use Trainer's distributed-aware save path. After the
training barrier, only global rank zero writes the final adapter, tokenizer,
and `run_summary.json`. A second barrier keeps worker shutdown coordinated.

Generated weights, caches, scheduler output, and runtime logs are excluded from
source control. The Git repository contains only code, configuration examples,
documentation, and sample data.

## Portability

Cluster-specific values stay outside the source files:

- Partition and QoS are supplied as `sbatch` options.
- Model, dataset, cache, and output paths are environment variables.
- The rendezvous host is derived from the Slurm allocation.
- The port and network interface filter can be overridden per cluster.
- Training and LoRA hyperparameters can be changed without editing Python.

## Extension points

The same launch pattern can support larger datasets, different compatible
base checkpoints, experiment tracking, evaluation jobs, and alternative LoRA
target modules. The generated `run_summary.json` provides a simple integration
point for downstream reporting and model registries.
