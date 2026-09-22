# Distributed QLoRA for GPT-OSS 120B

A multi-node training pipeline for parameter-efficient fine-tuning of GPT-OSS
120B with Unsloth, PyTorch Distributed, `torchrun`, and Slurm. The default
topology targets **24 NVIDIA H100 GPUs across three nodes**.

## Highlights

- 4-bit model loading with trainable LoRA adapters
- 24-process NCCL data parallelism across three Slurm nodes
- Deterministic rank and device assignment through `torchrun`
- Gradient checkpointing and configurable accumulation for memory efficiency
- Flexible JSONL ingestion for chat messages, role columns, or plain text
- Single-writer adapter export with a machine-readable run summary
- Standalone commands for adapter inference and 16-bit model export
- Environment-driven paths and hyperparameters for cluster portability

## Architecture

```text
Slurm allocation (3 nodes x 8 H100 GPUs)
                  |
                  v
       one srun task per node
                  |
                  v
       torchrun (8 workers/node)
                  |
                  v
      24-worker NCCL process group
                  |
                  v
   GPT-OSS 120B 4-bit + LoRA + SFTTrainer
                  |
                  v
      rank-zero adapter and run summary
```

Slurm starts one launcher on each node. Each launcher creates eight workers,
one per GPU, and all workers join a shared C10d rendezvous. The training entry
point verifies CUDA placement and distributed world size before loading the
model. Only global rank zero writes final artifacts.

See [docs/architecture.md](docs/architecture.md) for the launch sequence and
design decisions.

## Repository layout

| Path | Purpose |
| --- | --- |
| `train_lora_gptoss_unsloth.py` | Dataset preparation, LoRA setup, distributed training, and export |
| `run_sft_24gpus_gptoss.slurm` | Three-node Slurm job definition |
| `launch_torchrun.sh` | Per-node distributed launcher |
| `00_env.sh` | Portable environment and cache configuration |
| `infer_lora_local.py` | Adapter-based generation on one GPU |
| `merge_lora.py` | Optional 16-bit merged export |
| `examples/sample_train.jsonl` | Minimal examples of supported dataset schemas |
| `.env.example` | Configurable paths and hyperparameters |

## Requirements

- Linux cluster with Slurm
- 3 nodes with 8 NVIDIA H100 GPUs per node for the default topology
- CUDA-compatible PyTorch installation
- Python 3.10+
- Access to the GPT-OSS 120B base checkpoint

Create an environment and install the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Configuration

Copy the example configuration and customize the values for your environment:

```bash
cp .env.example .env
set -a
source .env
set +a
source ./00_env.sh
```

Key settings:

| Variable | Default | Description |
| --- | --- | --- |
| `MODEL_ID` | `unsloth/gpt-oss-120b-unsloth-bnb-4bit` | Base model ID or local checkpoint path |
| `DATASET` | `examples/sample_train.jsonl` | Training JSONL path |
| `OUT_DIR` | `runs/gptoss120b-lora` | Adapter and run-summary directory |
| `MAX_SEQ_LEN` | `2048` | Maximum sequence length |
| `BSZ` | `1` | Per-device micro-batch size |
| `GA` | `16` | Gradient accumulation steps |
| `LR` | `2e-4` | Learning rate |
| `MAX_STEPS` | `200` | Number of optimizer steps |
| `LORA_R` | `16` | LoRA rank |
| `LORA_ALPHA` | `16` | LoRA scaling factor |
| `LORA_TARGET_MODULES` | attention and MLP projections | Comma-separated target modules |

The effective global batch size is:

```text
world_size x BSZ x GA
```

With the default 24-worker topology, that is `24 x 1 x 16 = 384` sequences
per optimizer step before packing effects.

## Dataset format

The loader accepts several JSONL schemas. A chat-style row looks like this:

```json
{"messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"Add 7 + 8."},{"role":"assistant","content":"15"}]}
```

Role columns are also supported:

```json
{"system":"You are a helpful assistant.","user":"What is 12 + 23?","assistant":"35"}
```

Replace the included sample with the project dataset and point `DATASET` to
its location.

## Launch training

Submit the job from the repository root. Partition and QoS names are supplied
at submission time so the checked-in job file remains portable:

```bash
sbatch \
  --partition=<partition> \
  --qos=<qos> \
  run_sft_24gpus_gptoss.slurm
```

The output directory contains the LoRA adapter, tokenizer files, trainer
artifacts, and `run_summary.json`. The summary records the model ID, dataset
size, world size, effective global batch size, step count, and final metrics.

## Run inference

```bash
./run_infer.sh \
  --adapter ./runs/gptoss120b-lora \
  --prompt "Explain parameter-efficient fine-tuning in two sentences."
```

Use `--base-model` to select a local checkpoint or another compatible model
reference. Add `--local-files-only` for fully offline inference.

## Export merged weights

```bash
python merge_lora.py \
  --adapter ./runs/gptoss120b-lora \
  --output-dir ./runs/gptoss120b-merged
```

Adapter-only artifacts remain the smallest and most convenient format for
iteration. The merge command is available when a standalone 16-bit export is
needed.

## Validation checklist

For a production run, confirm that:

1. The startup banner reports `world_size=24`.
2. All workers join the same rendezvous endpoint.
3. `run_summary.json` records the expected dataset size and global batch size.
4. The saved adapter reloads with the same base-model revision.
5. Evaluation is performed on a held-out dataset aligned with the target use
   case.

## Engineering focus

This repository emphasizes reusable distributed-training infrastructure:
portable configuration, deterministic process placement, rank-aware artifact
ownership, and a clean separation between training, inference, and export.
