# Archived experiment notes

## Evidence reviewed

The downloaded bundle contained the training and launch scripts, a two-row toy
dataset, job `32825` stdout/stderr, and job `33613` inference stdout/stderr. It
did not include the resulting adapter, base-model weights, package lock file, or
a successful inference transcript.

## Job 32825

| Item | Observed evidence | Interpretation |
| --- | --- | --- |
| Allocation | 3 nodes, 8 H100 GPUs per node | 24 GPUs were allocated |
| Launcher | 3 node-level tasks, each starting 8 `torchrun` workers | 24 processes were launched |
| Model | `unsloth/gpt-oss-120b-unsloth-bnb-4bit` | Workers attempted to load the intended 4-bit base |
| Data | `toy.jsonl`, 2 rows | Pipeline smoke test only |
| Steps | 200 configured steps | Workers performed local training loops |
| Reported loss | approximately 0.1613 | Expected overfitting on two examples; not an evaluation metric |
| Trainer topology | `Data Parallel GPUs = 1` in worker banners | Logs do not establish one synchronized 24-way trainer |
| Saving | repeated saves to one shared path | Race/corruption risk; artifact provenance is ambiguous |
| Shutdown | process-group warnings and 300-second exit-barrier timeouts | Distributed completion was not clean |

Because the logs do not prove synchronized gradients and one authoritative
adapter, this job should not be described as a successful 24-GPU fine-tune.

## Inference job 33613

The inference job failed while resolving model configuration in offline mode.
The required files were not present in the selected local cache, and outbound
access was disabled. There is no generated answer in the captured stdout.

## Separate distributed validation

The broader cluster audit also found successful three-node, 24-GPU DDP jobs
that reached 1,000 synchronized steps with checkpoint/resume. Inspection of
their checkpoints showed a 131,712-parameter two-layer MLP, not GPT-OSS 120B.
Those jobs validate distributed-systems plumbing, but they are not evidence of
large-model fine-tuning.

## Changes made for the public repository

- Removed account-specific working directories and home-directory assumptions.
- Removed runtime package uninstallation from the environment script.
- Corrected the missing training-entrypoint mismatch.
- Stopped rewriting `CUDA_VISIBLE_DEVICES` inside each Python worker.
- Added explicit world-size and CUDA-placement validation.
- Restricted final adapter export to global rank zero.
- Removed in-training merge attempts; merging is now an explicit operation.
- Added portable CLI arguments for inference and merging.
- Excluded raw logs and generated artifacts from version control.
