"""Seed demo data for investor demo.

Creates realistic-looking experiments, metrics, prompts and datasets
so the Dashboard and all pages look populated during the demo.

Usage:
    python scripts/seed_demo_data.py
"""

import json
import math
import random
import uuid
from datetime import datetime, timedelta

from pulsar_ai.storage.database import get_database


def main() -> None:
    db = get_database()

    now = datetime.now()

    # ── Completed experiments ────────────────────────────────────
    experiments = [
        {
            "id": str(uuid.uuid4())[:8],
            "name": "llama3.2-1b-customer-support-sft",
            "status": "completed",
            "task": "sft",
            "model": "meta-llama/Llama-3.2-1B",
            "final_loss": 0.4823,
            "days_ago": 3,
            "epochs": 3,
            "lr": 5e-5,
            "batch_size": 4,
            "accuracy": 0.891,
        },
        {
            "id": str(uuid.uuid4())[:8],
            "name": "mistral-7b-code-review-dpo",
            "status": "completed",
            "task": "dpo",
            "model": "mistralai/Mistral-7B-v0.3",
            "final_loss": 0.3156,
            "days_ago": 2,
            "epochs": 2,
            "lr": 2e-5,
            "batch_size": 2,
            "accuracy": 0.923,
        },
        {
            "id": str(uuid.uuid4())[:8],
            "name": "qwen2.5-3b-sql-generation",
            "status": "completed",
            "task": "sft",
            "model": "Qwen/Qwen2.5-3B",
            "final_loss": 0.5241,
            "days_ago": 1,
            "epochs": 5,
            "lr": 3e-5,
            "batch_size": 8,
            "accuracy": 0.856,
        },
        {
            "id": str(uuid.uuid4())[:8],
            "name": "phi3-mini-intent-classifier",
            "status": "completed",
            "task": "sft",
            "model": "microsoft/Phi-3-mini-4k-instruct",
            "final_loss": 0.2987,
            "days_ago": 1,
            "epochs": 4,
            "lr": 1e-4,
            "batch_size": 16,
            "accuracy": 0.947,
        },
        {
            "id": str(uuid.uuid4())[:8],
            "name": "gemma2-2b-summarization",
            "status": "running",
            "task": "sft",
            "model": "google/gemma-2-2b-it",
            "final_loss": None,
            "days_ago": 0,
            "epochs": 3,
            "lr": 5e-5,
            "batch_size": 4,
            "accuracy": None,
        },
    ]

    for exp in experiments:
        created_at = (now - timedelta(days=exp["days_ago"])).isoformat()
        completed_at = (
            (now - timedelta(days=exp["days_ago"]) + timedelta(hours=random.randint(1, 4))).isoformat()
            if exp["status"] == "completed"
            else None
        )

        config = {
            "model": {"name": exp["model"]},
            "training": {
                "epochs": exp["epochs"],
                "learning_rate": exp["lr"],
                "batch_size": exp["batch_size"],
                "gradient_accumulation_steps": 4,
                "max_seq_length": 2048,
                "warmup_steps": 50,
                "optimizer": "adamw_torch",
                "seed": 42,
            },
            "lora": {
                "r": 16,
                "alpha": 32,
                "dropout": 0.05,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            },
        }

        artifacts = {
            "hyperparameters": config["training"],
            "lora": config["lora"],
            "strategy": "qlora_4bit",
            "quantization": {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": "bfloat16",
            },
            "hardware": {
                "gpu_name": "NVIDIA RTX 4090",
                "vram_gb": 24,
                "bf16": True,
            },
            "dataset": {
                "path": f"data/{exp['name'].split('-')[-1]}_train.jsonl",
                "format": "alpaca",
                "test_size": 0.1,
            },
            "trainable_params": random.randint(5_000_000, 20_000_000),
            "total_params": random.randint(1_000_000_000, 7_000_000_000),
            "trainable_pct": round(random.uniform(0.5, 2.5), 2),
            "adapter_size_mb": round(random.uniform(15.0, 45.0), 1),
            "training_duration_min": round(random.uniform(12.0, 90.0), 1),
        }

        eval_results = None
        if exp["accuracy"] is not None:
            eval_results = {
                "overall_accuracy": exp["accuracy"],
                "json_parse_rate": round(random.uniform(0.95, 1.0), 3),
                "f1_weighted": {
                    "f1": round(exp["accuracy"] - random.uniform(0.01, 0.03), 3),
                    "precision": round(exp["accuracy"] + random.uniform(0.0, 0.02), 3),
                    "recall": round(exp["accuracy"] - random.uniform(0.0, 0.02), 3),
                },
            }

        db.execute(
            """
            INSERT OR REPLACE INTO experiments
                (id, name, status, task, model, dataset_id, config,
                 created_at, last_update_at, completed_at, final_loss,
                 eval_results, artifacts)
            VALUES (?, ?, ?, ?, ?, '', ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                exp["id"],
                exp["name"],
                exp["status"],
                exp["task"],
                exp["model"],
                json.dumps(config),
                created_at,
                completed_at or created_at,
                completed_at,
                exp["final_loss"],
                json.dumps(eval_results) if eval_results else None,
                json.dumps(artifacts),
            ),
        )

        # Generate training history (loss curve)
        total_steps = exp["epochs"] * random.randint(50, 150)
        if exp["status"] == "running":
            total_steps = int(total_steps * 0.6)  # partial

        history = []
        initial_loss = random.uniform(1.8, 2.5)
        for step in range(total_steps):
            progress = step / max(total_steps - 1, 1)
            # Realistic loss curve: exponential decay + noise
            loss = initial_loss * math.exp(-3.0 * progress) + random.gauss(0, 0.02)
            loss = max(0.15, loss)  # floor
            epoch = (step / (total_steps / exp["epochs"]))
            history.append({
                "step": step,
                "epoch": round(epoch, 2),
                "loss": round(loss, 4),
            })

        # Store as training_history in experiment row
        db.execute(
            "UPDATE experiments SET artifacts = ? WHERE id = ?",
            (
                json.dumps({**artifacts, "training_history_ref": True}),
                exp["id"],
            ),
        )

        # Store metrics in experiment_metrics table
        for m in history:
            db.execute(
                """
                INSERT INTO experiment_metrics (experiment_id, data, recorded_at)
                VALUES (?, ?, ?)
                """,
                (exp["id"], json.dumps(m), created_at),
            )

    # ── Prompts ──────────────────────────────────────────────────
    prompts = [
        ("Customer Support Agent", "system_prompt for customer support classification", "classification,support"),
        ("SQL Query Generator", "Generate SQL from natural language", "sql,code-gen"),
        ("Code Review Assistant", "Review pull requests and suggest improvements", "code-review,qa"),
    ]

    for name, desc, tags in prompts:
        pid = str(uuid.uuid4())[:8]
        ts = (now - timedelta(days=random.randint(1, 5))).isoformat()
        db.execute(
            """
            INSERT OR REPLACE INTO prompts
                (id, name, description, current_version, tags, created_at, updated_at)
            VALUES (?, ?, ?, 1, ?, ?, ?)
            """,
            (pid, name, desc, json.dumps(tags.split(",")), ts, ts),
        )
        db.execute(
            """
            INSERT OR REPLACE INTO prompt_versions
                (prompt_id, version, system_prompt, variables, model, parameters, created_at)
            VALUES (?, 1, ?, '[]', 'gpt-4o-mini', '{}', ?)
            """,
            (pid, f"You are a helpful {name.lower()}. {desc}", ts),
        )

    # ── Workflows ────────────────────────────────────────────────
    workflows = [
        "Data Prep → SFT → Eval Pipeline",
        "Weekly Retrain Workflow",
    ]
    for wf_name in workflows:
        wid = str(uuid.uuid4())[:8]
        ts = (now - timedelta(days=random.randint(2, 7))).isoformat()
        db.execute(
            """
            INSERT OR REPLACE INTO workflows
                (id, name, nodes, edges, schema_version, created_at, updated_at, run_count)
            VALUES (?, ?, '[]', '[]', 2, ?, ?, ?)
            """,
            (wid, wf_name, ts, ts, random.randint(1, 5)),
        )

    db.commit()
    print(f"Seeded {len(experiments)} experiments, {len(prompts)} prompts, {len(workflows)} workflows")
    print("Dashboard should now show populated data!")


if __name__ == "__main__":
    main()
