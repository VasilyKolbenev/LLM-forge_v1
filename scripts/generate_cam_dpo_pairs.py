"""Generate DPO preference pairs from CAM intent classification dataset.

Creates (prompt, chosen, rejected) triplets with hard negatives:
- Confusable domain swaps (semantically similar domains)
- Same domain, wrong skill (hardest to distinguish)
- Partial JSON errors (teaches formatting)
- Random domain (easy baseline)

Usage:
    python scripts/generate_cam_dpo_pairs.py \
        --input data/cam_intents.csv \
        --output data/cam_dpo_pairs.jsonl \
        --num-negatives 3
"""

import argparse
import json
import logging
import random
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Semantically confusable domain pairs (hard negatives)
CONFUSABLE_DOMAINS: dict[str, list[str]] = {
    "HOUSE": ["PAYMENTS", "UTILIZER"],      # housing bills ↔ payments, waste
    "PAYMENTS": ["HOUSE", "MOSBILET"],       # payments ↔ utility bills, tickets
    "MOSBILET": ["PAYMENTS"],                # ticket purchase ↔ payment
    "UTILIZER": ["HOUSE"],                   # waste ↔ housing
    "BIZ_ASSIST": ["PTNT"],                  # business ↔ patents
    "PTNT": ["BIZ_ASSIST"],                  # patents ↔ business
    "DNEVNIK": ["BIZ_ASSIST"],               # school ↔ consulting
    "BOLTALKA": ["EXIT"],                    # chitchat ↔ goodbye
    "EXIT": ["BOLTALKA"],                    # goodbye ↔ chitchat
}


def load_taxonomy(csv_path: str) -> dict[str, list[str]]:
    """Extract domain->skills mapping from dataset.

    Args:
        csv_path: Path to CAM intents CSV.

    Returns:
        Dict mapping domain to list of skills.
    """
    df = pd.read_csv(csv_path)
    taxonomy: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        domain = row["domain"]
        skill = row["skill"]
        if domain not in taxonomy:
            taxonomy[domain] = []
        if skill not in taxonomy[domain]:
            taxonomy[domain].append(skill)
    return taxonomy


def make_chosen(domain: str, skill: str) -> str:
    """Create the correct (chosen) response."""
    return json.dumps(
        {"domain": domain, "skill": skill, "confidence": "high"},
        ensure_ascii=False,
    )


def make_hard_negative(
    true_domain: str,
    true_skill: str,
    taxonomy: dict[str, list[str]],
    strategy: str,
    rng: random.Random,
) -> str | None:
    """Create a rejected response using a specific strategy.

    Args:
        true_domain: Correct domain.
        true_skill: Correct skill.
        taxonomy: Domain->skills mapping.
        strategy: One of 'confusable', 'wrong_skill', 'random_domain', 'bad_format'.
        rng: Random number generator.

    Returns:
        JSON string with wrong classification, or None if strategy not applicable.
    """
    if strategy == "confusable":
        # Hard negative: swap to a semantically similar domain
        confusable = CONFUSABLE_DOMAINS.get(true_domain, [])
        valid = [d for d in confusable if d in taxonomy]
        if valid:
            wrong_domain = rng.choice(valid)
            wrong_skill = rng.choice(taxonomy[wrong_domain])
            return json.dumps(
                {"domain": wrong_domain, "skill": wrong_skill, "confidence": "high"},
                ensure_ascii=False,
            )
        return None

    if strategy == "wrong_skill":
        # Same domain, wrong skill (very hard to distinguish)
        other_skills = [s for s in taxonomy[true_domain] if s != true_skill]
        if other_skills:
            return json.dumps(
                {"domain": true_domain, "skill": rng.choice(other_skills),
                 "confidence": "high"},
                ensure_ascii=False,
            )
        return None

    if strategy == "random_domain":
        # Easy negative: completely wrong domain
        wrong_domains = [d for d in taxonomy if d != true_domain]
        if wrong_domains:
            wrong_domain = rng.choice(wrong_domains)
            wrong_skill = rng.choice(taxonomy[wrong_domain])
            return json.dumps(
                {"domain": wrong_domain, "skill": wrong_skill, "confidence": "medium"},
                ensure_ascii=False,
            )
        return None

    if strategy == "bad_format":
        # Teaches proper JSON formatting and field completeness
        variant = rng.choice(["missing_field", "wrong_value", "swapped"])
        if variant == "missing_field":
            return json.dumps(
                {"domain": true_domain, "skill": true_skill},
                ensure_ascii=False,
            )
        if variant == "wrong_value":
            return json.dumps(
                {"domain": true_domain.lower(), "skill": true_skill,
                 "confidence": "high"},
                ensure_ascii=False,
            )
        # swapped: domain and skill values reversed
        return json.dumps(
            {"domain": true_skill, "skill": true_domain, "confidence": "high"},
            ensure_ascii=False,
        )

    return None


def generate_pairs(
    csv_path: str,
    num_negatives: int = 3,
    seed: int = 42,
) -> list[dict[str, str]]:
    """Generate DPO pairs with diverse hard negatives.

    Each prompt gets `num_negatives` rejected responses using different strategies.
    Strategy distribution: confusable (35%), wrong_skill (30%), random (20%), bad_format (15%).

    Args:
        csv_path: Path to CAM intents CSV.
        num_negatives: Number of rejected responses per prompt.
        seed: Random seed.

    Returns:
        List of {prompt, chosen, rejected} dicts.
    """
    df = pd.read_csv(csv_path)
    taxonomy = load_taxonomy(csv_path)
    rng = random.Random(seed)

    # Weighted strategy pool
    strategy_weights = [
        ("confusable", 0.35),
        ("wrong_skill", 0.30),
        ("random_domain", 0.20),
        ("bad_format", 0.15),
    ]
    strategies = [s for s, _ in strategy_weights]
    weights = [w for _, w in strategy_weights]

    pairs = []
    strategy_counts: dict[str, int] = {s: 0 for s in strategies}

    for _, row in df.iterrows():
        phrase = row["phrase"]
        domain = row["domain"]
        skill = row["skill"]
        chosen = make_chosen(domain, skill)

        used_rejections: set[str] = set()
        generated = 0

        # Try to generate diverse negatives
        attempts = 0
        while generated < num_negatives and attempts < num_negatives * 5:
            attempts += 1
            strategy = rng.choices(strategies, weights=weights, k=1)[0]
            rejected = make_hard_negative(
                domain, skill, taxonomy, strategy, rng
            )

            if rejected is None or rejected == chosen or rejected in used_rejections:
                continue

            used_rejections.add(rejected)
            pairs.append({
                "prompt": phrase,
                "chosen": chosen,
                "rejected": rejected,
            })
            strategy_counts[strategy] += 1
            generated += 1

    rng.shuffle(pairs)

    logger.info("Strategy distribution:")
    for s, c in strategy_counts.items():
        logger.info("  %s: %d (%.1f%%)", s, c, c / len(pairs) * 100 if pairs else 0)

    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CAM DPO pairs")
    parser.add_argument(
        "--input", default="data/cam_intents.csv", help="Input CSV path"
    )
    parser.add_argument(
        "--output", default="data/cam_dpo_pairs.jsonl", help="Output JSONL path"
    )
    parser.add_argument(
        "--num-negatives", type=int, default=3, help="Negatives per sample"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    pairs = generate_pairs(args.input, args.num_negatives, args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    logger.info("Generated %d DPO pairs -> %s", len(pairs), output_path)

    taxonomy = load_taxonomy(args.input)
    logger.info(
        "Taxonomy: %d domains, %d total skills",
        len(taxonomy),
        sum(len(v) for v in taxonomy.values()),
    )


if __name__ == "__main__":
    main()
