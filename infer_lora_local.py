#!/usr/bin/env python3
"""Load a GPT-OSS base model plus a LoRA adapter and generate one response."""

from __future__ import annotations

import argparse
import os

import torch
from peft import PeftModel
from transformers import TextStreamer
from unsloth import FastLanguageModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-model",
        default=os.environ.get(
            "MODEL_ID", "unsloth/gpt-oss-120b-unsloth-bnb-4bit"
        ),
    )
    parser.add_argument(
        "--adapter",
        default=os.environ.get("OUT_DIR", "runs/gptoss120b-lora"),
    )
    parser.add_argument(
        "--prompt",
        default="Give me a two-sentence summary of LoRA fine-tuning.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")

    torch.set_grad_enabled(False)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=True,
        local_files_only=args.local_files_only,
        trust_remote_code=True,
        device_map={"": 0},
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    FastLanguageModel.for_inference(model)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": args.prompt},
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        prompt = f"User: {args.prompt}\nAssistant:"

    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        streamer=streamer,
    )


if __name__ == "__main__":
    main()
