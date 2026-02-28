"""SFT (Supervised Fine-Tuning) trainer.

Supports two backends:
- Unsloth: 2-5x faster on single GPU (recommended for ≤24GB VRAM)
- HuggingFace SFTTrainer: multi-GPU via Accelerate
"""

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _get_lora_params(config: dict) -> dict:
    """Resolve LoRA parameters from config.

    Reads from 'lora' section first, falls back to top-level keys.

    Args:
        config: Full config dict.

    Returns:
        Dict with r, lora_alpha, lora_dropout, target_modules, bias.
    """
    lora = config.get("lora", {})
    model_config = config.get("model", {})
    default_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    return {
        "r": lora.get("r", config.get("lora_r", 16)),
        "lora_alpha": lora.get("lora_alpha", config.get("lora_alpha", 32)),
        "lora_dropout": lora.get("lora_dropout", config.get("lora_dropout", 0)),
        "target_modules": lora.get(
            "target_modules",
            model_config.get("lora_target_modules", default_modules),
        ),
        "bias": lora.get("bias", "none"),
    }


def train_sft(config: dict) -> dict:
    """Run SFT training based on config.

    Auto-selects backend (Unsloth vs HF) based on strategy.

    Args:
        config: Fully resolved config dict.

    Returns:
        Dict with training results (loss, steps, output_dir).
    """
    strategy = config.get("_detected_strategy", config.get("strategy", "qlora"))
    use_unsloth = config.get("use_unsloth", strategy in ("qlora", "lora"))
    fsdp = config.get("fsdp_enabled", False)

    if use_unsloth and not fsdp:
        return _train_sft_unsloth(config)
    else:
        return _train_sft_hf(config)


def _train_sft_unsloth(config: dict) -> dict:
    """SFT training with Unsloth backend (single GPU).

    Args:
        config: Resolved config dict.

    Returns:
        Training results dict.
    """
    import torch
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import Dataset

    model_config = config.get("model", {})
    training_config = config.get("training", {})
    output_dir = config.get("output", {}).get("dir", "./outputs/sft")

    logger.info("Loading model via Unsloth: %s", model_config.get("name"))
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_config.get("name"),
        max_seq_length=training_config.get("max_seq_length", 1024),
        dtype=None,
        load_in_4bit=config.get("load_in_4bit", True),
    )

    # Apply LoRA if configured
    if config.get("use_lora", True):
        lora_params = _get_lora_params(config)
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_params["r"],
            target_modules=lora_params["target_modules"],
            lora_alpha=lora_params["lora_alpha"],
            lora_dropout=lora_params["lora_dropout"],
            bias=lora_params["bias"],
            use_gradient_checkpointing=(
                "unsloth" if config.get("gradient_checkpointing") else False
            ),
            random_state=training_config.get("seed", 42),
        )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("Trainable: %s / %s (%.2f%%)", f"{trainable:,}", f"{total:,}", trainable / total * 100)

    # Load dataset
    train_dataset = _load_train_dataset(config, tokenizer)

    # Training arguments
    args = TrainingArguments(
        per_device_train_batch_size=training_config.get("batch_size", 1),
        gradient_accumulation_steps=training_config.get("gradient_accumulation", 16),
        warmup_steps=training_config.get("warmup_steps", 20),
        num_train_epochs=training_config.get("epochs", 3),
        learning_rate=training_config.get("learning_rate", 2e-4),
        bf16=config.get("_hardware", {}).get("bf16_supported", True),
        logging_steps=training_config.get("logging_steps", 20),
        save_steps=training_config.get("save_steps", 200),
        save_total_limit=training_config.get("save_total_limit", 2),
        optim=training_config.get("optimizer", "adamw_8bit"),
        output_dir=output_dir,
        seed=training_config.get("seed", 42),
        report_to=config.get("logging", {}).get("report_to", "none"),
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=training_config.get("max_seq_length", 512),
        packing=training_config.get("packing", True),
        args=args,
    )

    resume_checkpoint = config.get("resume_from_checkpoint")
    if resume_checkpoint:
        logger.info("Resuming from checkpoint: %s", resume_checkpoint)

    logger.info("Starting SFT training...")
    stats = trainer.train(resume_from_checkpoint=resume_checkpoint)

    # Save adapter
    adapter_dir = str(Path(output_dir) / "lora")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    logger.info("LoRA adapter saved to %s", adapter_dir)

    vram_peak = torch.cuda.max_memory_allocated() / (1024**3)
    return {
        "training_loss": stats.training_loss,
        "global_steps": stats.global_step,
        "vram_peak_gb": round(vram_peak, 2),
        "output_dir": output_dir,
        "adapter_dir": adapter_dir,
    }


def _train_sft_hf(config: dict) -> dict:
    """SFT training with HuggingFace SFTTrainer (multi-GPU ready).

    Args:
        config: Resolved config dict.

    Returns:
        Training results dict.
    """
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer

    model_config = config.get("model", {})
    training_config = config.get("training", {})
    output_dir = config.get("output", {}).get("dir", "./outputs/sft")

    model_name = model_config.get("name")
    logger.info("Loading model via HF Transformers: %s", model_name)

    # Quantization config
    bnb_config = None
    if config.get("load_in_4bit"):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=getattr(
                torch, config.get("bnb_4bit_compute_dtype", "bfloat16")
            ),
            bnb_4bit_use_double_quant=config.get(
                "bnb_4bit_use_double_quant", True
            ),
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto" if not config.get("fsdp_enabled") else None,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Apply LoRA
    if config.get("use_lora", True):
        lora_params = _get_lora_params(config)
        lora_config = LoraConfig(
            r=lora_params["r"],
            lora_alpha=lora_params["lora_alpha"],
            lora_dropout=lora_params["lora_dropout"],
            target_modules=lora_params["target_modules"],
            bias=lora_params["bias"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    train_dataset = _load_train_dataset(config, tokenizer)

    args = TrainingArguments(
        per_device_train_batch_size=training_config.get("batch_size", 2),
        gradient_accumulation_steps=training_config.get("gradient_accumulation", 8),
        warmup_steps=training_config.get("warmup_steps", 20),
        num_train_epochs=training_config.get("epochs", 3),
        learning_rate=training_config.get("learning_rate", 2e-4),
        bf16=config.get("_hardware", {}).get("bf16_supported", True),
        logging_steps=training_config.get("logging_steps", 20),
        save_steps=training_config.get("save_steps", 200),
        save_total_limit=2,
        optim=training_config.get("optimizer", "adamw_8bit"),
        output_dir=output_dir,
        seed=training_config.get("seed", 42),
        report_to=config.get("logging", {}).get("report_to", "none"),
        gradient_checkpointing=config.get("gradient_checkpointing", True),
        # FSDP settings
        fsdp=config.get("fsdp_sharding_strategy") if config.get("fsdp_enabled") else "",
        fsdp_config={
            "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            "fsdp_cpu_offload": config.get("fsdp_cpu_offload", False),
        } if config.get("fsdp_enabled") else None,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=training_config.get("max_seq_length", 512),
        packing=training_config.get("packing", True),
        args=args,
    )

    resume_checkpoint = config.get("resume_from_checkpoint")
    if resume_checkpoint:
        logger.info("Resuming from checkpoint: %s", resume_checkpoint)

    logger.info("Starting SFT training (HF backend)...")
    stats = trainer.train(resume_from_checkpoint=resume_checkpoint)

    adapter_dir = str(Path(output_dir) / "lora")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    logger.info("Model saved to %s", adapter_dir)

    return {
        "training_loss": stats.training_loss,
        "global_steps": stats.global_step,
        "output_dir": output_dir,
        "adapter_dir": adapter_dir,
    }


def _load_train_dataset(config: dict, tokenizer: Any) -> Any:
    """Load and format training dataset from config.

    Args:
        config: Full config dict.
        tokenizer: Tokenizer for chat template.

    Returns:
        HuggingFace Dataset with "text" field.
    """
    from llm_forge.data.loader import load_dataset_from_config
    from llm_forge.data.formatter import (
        build_chat_examples,
        apply_chat_template,
        load_system_prompt,
    )
    from llm_forge.data.splitter import split_dataset

    df = load_dataset_from_config(config)
    ds_config = config.get("dataset", {})

    # Split
    splits = split_dataset(
        df,
        test_size=ds_config.get("test_size", 0.15),
        stratify_column=ds_config.get("stratify_column"),
        seed=config.get("training", {}).get("seed", 42),
    )
    train_df = splits["train"]

    # System prompt
    system_prompt = ""
    prompt_file = ds_config.get("system_prompt_file")
    if prompt_file:
        system_prompt = load_system_prompt(prompt_file)
    elif ds_config.get("system_prompt"):
        system_prompt = ds_config["system_prompt"]

    # Build chat examples
    examples = build_chat_examples(
        train_df,
        system_prompt=system_prompt,
        text_column=ds_config.get("text_column", "phrase"),
        label_columns=ds_config.get("label_columns", ["label"]),
        output_format=ds_config.get("output_format", "json"),
    )

    return apply_chat_template(examples, tokenizer)
