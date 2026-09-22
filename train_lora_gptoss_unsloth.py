#!/usr/bin/env python3
"""Environment-driven GPT-OSS LoRA training entry point.

The launcher starts one process per GPU. This module deliberately does not
rewrite CUDA_VISIBLE_DEVICES: torchrun owns process placement.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import unsloth  # noqa: F401 - Unsloth must patch libraries before TRL imports.
from datasets import Dataset, load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel


def env(name: str, default: Any, cast: type = str) -> Any:
    value = os.environ.get(name)
    return default if value is None else cast(value)


MODEL_ID = env("MODEL_ID", "unsloth/gpt-oss-120b-unsloth-bnb-4bit")
DATASET_PATH = Path(env("DATASET", "examples/sample_train.jsonl")).expanduser().resolve()
OUTPUT_DIR = Path(env("OUT_DIR", "runs/gptoss120b-lora")).expanduser().resolve()
MAX_SEQ_LEN = env("MAX_SEQ_LEN", 2048, int)
MICRO_BATCH_SIZE = env("BSZ", 1, int)
GRAD_ACCUM = env("GA", 16, int)
LEARNING_RATE = env("LR", 2e-4, float)
MAX_STEPS = env("MAX_STEPS", 200, int)
SAVE_STEPS = env("SAVE_STEPS", 200, int)
LOG_STEPS = env("LOG_STEPS", 5, int)
LORA_R = env("LORA_R", 16, int)
LORA_ALPHA = env("LORA_ALPHA", 16, int)
LORA_DROPOUT = env("LORA_DROPOUT", 0.0, float)
SEED = env("SEED", 42, int)
NUM_WORKERS = env("NUM_WORKERS", 2, int)
TARGET_MODULES = [
    item.strip()
    for item in env(
        "LORA_TARGET_MODULES",
        "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    ).split(",")
    if item.strip()
]


def distributed_context() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training configuration")
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank}, but only {torch.cuda.device_count()} GPUs are visible"
        )

    torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    if world_size > 1 and dist.get_world_size() != world_size:
        raise RuntimeError(
            f"Process-group world size {dist.get_world_size()} != WORLD_SIZE {world_size}"
        )
    return rank, local_rank, world_size


def row_to_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    messages = row.get("messages") or row.get("conversations")
    if isinstance(messages, list):
        normalized = []
        for message in messages:
            if isinstance(message, dict) and "role" in message and "content" in message:
                normalized.append(
                    {"role": str(message["role"]), "content": str(message["content"])}
                )
        if normalized:
            return normalized

    normalized = []
    for role in ("system", "user", "assistant"):
        value = row.get(role)
        if value:
            normalized.append({"role": role, "content": str(value)})
    if normalized:
        return normalized

    if row.get("text"):
        return [{"role": "user", "content": str(row["text"])}]
    return []


def load_training_data(path: Path) -> Dataset:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    dataset = load_dataset("json", data_files=str(path), split="train")
    dataset = dataset.filter(lambda row: bool(row_to_messages(row)))
    if len(dataset) == 0:
        raise RuntimeError("Dataset contains no usable conversations")
    return dataset


def main() -> None:
    rank, local_rank, world_size = distributed_context()
    is_main = rank == 0
    random.seed(SEED)
    torch.manual_seed(SEED)

    if is_main:
        print("=== GPT-OSS LoRA training ===")
        print(f"model={MODEL_ID}")
        print(f"dataset={DATASET_PATH}")
        print(f"output={OUTPUT_DIR}")
        print(
            f"world_size={world_size} micro_batch={MICRO_BATCH_SIZE} "
            f"gradient_accumulation={GRAD_ACCUM} "
            f"global_batch={world_size * MICRO_BATCH_SIZE * GRAD_ACCUM}"
        )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=MAX_SEQ_LEN,
        dtype=torch.bfloat16,
        load_in_4bit=True,
        # Leave placement to the current CUDA device and Trainer/DDP. Passing a
        # device map here can make Accelerate treat the model as model-parallel.
        device_map=None,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )

    dataset = load_training_data(DATASET_PATH)
    if is_main and len(dataset) < world_size:
        print(
            f"[data] {len(dataset)} rows across {world_size} workers; "
            "use a full training dataset for a distributed run."
        )

    def formatting_func(row: dict[str, Any]) -> list[str]:
        messages = row_to_messages(row)
        if not messages:
            return []
        return [
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        max_steps=MAX_STEPS,
        logging_steps=LOG_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        dataloader_num_workers=NUM_WORKERS,
        ddp_find_unused_parameters=False,
        local_rank=local_rank,
        save_on_each_node=False,
        report_to=[],
        seed=SEED,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        formatting_func=formatting_func,
        max_seq_length=MAX_SEQ_LEN,
        packing=True,
        args=training_args,
    )
    result = trainer.train()

    if dist.is_initialized():
        dist.barrier()

    if is_main:
        trainer.save_model(str(OUTPUT_DIR))
        tokenizer.save_pretrained(str(OUTPUT_DIR))
        summary = {
            "model_id": MODEL_ID,
            "dataset_rows": len(dataset),
            "world_size": world_size,
            "max_steps": MAX_STEPS,
            "global_batch_size": world_size * MICRO_BATCH_SIZE * GRAD_ACCUM,
            "metrics": result.metrics,
        }
        (OUTPUT_DIR / "run_summary.json").write_text(
            json.dumps(summary, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        print(f"Saved adapter and run summary to {OUTPUT_DIR}")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
