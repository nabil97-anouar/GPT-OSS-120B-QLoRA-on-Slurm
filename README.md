# GPT-OSS 120B QLoRA on Slurm

An experimental, environment-driven workflow for loading a 4-bit GPT-OSS 120B
checkpoint, attaching LoRA adapters with Unsloth, and launching supervised
fine-tuning across a Slurm allocation.

This repository is intentionally evidence-conscious: it contains reusable code
and a transparent account of the archived experiment, without presenting an
inconclusive run as a validated 24-GPU fine-tune.

## What this project demonstrates

- Slurm orchestration for three nodes with eight H100 GPUs per node.
- One `torchrun` worker per GPU with a shared multi-node rendezvous.
- Environment-configurable 4-bit model loading and LoRA injection.
- Chat-aware JSONL ingestion for `messages`, role columns, or plain text.
- Global-rank-only adapter export and a machine-readable run summary.
- Separate inference and 16-bit merge utilities.
- Defensive checks for CUDA placement, process-group size, and undersized data.

## Experiment status

The archived job `32825` allocated 24 GPUs and launched 24 workers across three
nodes. Every worker loaded the 4-bit checkpoint and progressed through the
configured 200 steps on a two-row toy dataset. The logs reported a final loss of
approximately `0.1613`.

That run is **not treated as proof of a correct synchronized 24-way fine-tune**:

- the trainer banner reported one data-parallel GPU per worker;
- workers progressed and terminated at substantially different times;
- multiple workers attempted to save into the same output directory;
- distributed exit barriers timed out; and
- the recorded inference job failed because the required base-model files were
  unavailable in the offline cache.

The result is best understood as a large-model loading and LoRA-training
experiment that exposed distributed-integration and artifact-management issues.
The code in this repository addresses those issues, but the corrected path has
not yet been rerun and benchmarked.

See [docs/experiment-notes.md](docs/experiment-notes.md) for the evidence table
and interpretation.

## Launch design

```text
Slurm allocation: 3 nodes x 8 GPUs
        |
        +-- one srun task per node
                |
                +-- torchrun: 8 workers per node
                        |
                        +-- 24-worker NCCL process group
                                |
                                +-- SFTTrainer + LoRA
```

Each process selects `LOCAL_RANK` without rewriting `CUDA_VISIBLE_DEVICES`.
Before model loading, the training entry point verifies that the NCCL process
group matches `WORLD_SIZE`. Only global rank zero writes the final adapter and
`run_summary.json`.

## Repository contents

| File | Purpose |
| --- | --- |
| `train_lora_gptoss_unsloth.py` | Training, dataset normalization, distributed validation, and saving |
| `launch_torchrun.sh` | Per-node `torchrun` launcher |
| `run_sft_24gpus_gptoss.slurm` | Example 3-node/24-GPU Slurm submission |
| `00_env.sh` | Portable environment defaults |
| `infer_lora_local.py` | Single-GPU adapter inference |
| `merge_lora.py` | Optional Unsloth 16-bit merged export |
| `toy.jsonl` | Two-row pipeline smoke-test dataset |

Raw scheduler logs are deliberately excluded by `.gitignore` because they can
contain usernames, filesystem paths, internal hostnames, and private IPs.

## Installation

The workflow requires Linux, CUDA, and a GPU with enough memory for the selected
quantized model. Install the CUDA-compatible PyTorch build recommended for the
cluster, then install the Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For reproducible research, capture the working environment after validation:

```bash
python -m pip freeze > requirements-lock.txt
```

No lock file is included because the archived environment metadata was not part
of the downloaded evidence bundle.

## Configuration

All important values can be overridden through environment variables. Start
with the example:

```bash
cp .env.example .env
set -a
source .env
set +a
source ./00_env.sh
```

The default `toy.jsonl` contains only two arithmetic conversations. It is for
pipeline validation, not model-quality training.

## Running on Slurm

Adjust the partition, QoS, time limit, GPU type, and CPU allocation for the
target cluster. Supply cluster-specific partition and QoS values as `sbatch`
flags or add them to a private copy of the submission file.

```bash
sbatch --partition=<partition> --qos=<qos> run_sft_24gpus_gptoss.slurm
```

A meaningful run should use a dataset much larger than the number of workers
and should verify all of the following before its results are reported:

1. `run_summary.json` records `world_size: 24`.
2. Only global rank zero writes the final adapter.
3. All ranks reach the final barrier without a timeout.
4. The adapter reloads against the exact base-model revision.
5. A held-out evaluation or qualitative inference succeeds.

## Inference

After a validated adapter has been produced:

```bash
./run_infer.sh \
  --adapter ./runs/gptoss120b-lora \
  --prompt "Explain parameter-efficient fine-tuning in two sentences."
```

Add `--local-files-only` only when the complete base checkpoint is already in
the local Hugging Face cache.

## Merging

Merging a 120B adapter into 16-bit weights requires substantial GPU memory,
host memory, and disk space. Adapter-only publication is usually more practical.

```bash
python merge_lora.py \
  --adapter ./runs/gptoss120b-lora \
  --output-dir ./runs/gptoss120b-merged
```

Treat a merged export as valid only after it reloads successfully and produces
an inference result consistent with the adapter-loaded model.

## Scope and limitations

- No model weights or adapters are included.
- No successful inference output is claimed from the archived run.
- The two-example loss is an overfitting signal, not a quality metric.
- This code has been hardened from the archived scripts but has not been rerun
  in the original 24-GPU environment.
- Cluster networking and scheduler policies vary; NCCL and Slurm settings may
  require local adjustment.
