"""Clean DPO pairs dataset: remove noise, generate quality hard negatives.

Identifies and fixes three types of problematic pairs:
1. Identical chosen/rejected (zero gradient signal)
2. Case-only difference (trivial formatting signal)
3. Swapped domain↔skill fields (structural not semantic signal)

Replaces them with semantically confusable hard negatives.

Usage:
    python scripts/clean_dpo_pairs.py --input data/cam_dpo_pairs.jsonl --output data/cam_dpo_pairs_clean.jsonl
"""

import argparse
import json
import logging
import random
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Hard negative mappings: for each (domain, skill), list semantically confusable alternatives
# These represent plausible misclassifications that teach the model real distinctions
HARD_NEGATIVES: dict[tuple[str, str], list[tuple[str, str]]] = {
    # HOUSE domain
    ("HOUSE", "repair_request"): [
        ("HOUSE", "neighbor_complaint"),  # both about housing problems
        ("HOUSE", "utility_bill"),        # maintenance vs billing
        ("UTILIZER", "waste_collection"), # infrastructure issue
    ],
    ("HOUSE", "utility_bill"): [
        ("PAYMENTS", "payment_status"),   # both about money/bills
        ("PAYMENTS", "payment_method"),   # paying related
        ("HOUSE", "repair_request"),      # both housing
    ],
    ("HOUSE", "neighbor_complaint"): [
        ("HOUSE", "repair_request"),      # both housing issues
        ("UTILIZER", "waste_collection"), # mess/disorder
        ("HOUSE", "utility_bill"),        # both housing
    ],
    # MOSBILET domain
    ("MOSBILET", "ticket_purchase"): [
        ("MOSBILET", "ticket_refund"),    # both about tickets
        ("PAYMENTS", "payment_method"),   # buying involves paying
        ("DNEVNIK", "schedule"),          # events/schedule
    ],
    ("MOSBILET", "ticket_refund"): [
        ("MOSBILET", "ticket_purchase"),  # both tickets
        ("PAYMENTS", "payment_status"),   # refund = payment issue
        ("PAYMENTS", "payment_method"),   # financial
    ],
    # UTILIZER domain
    ("UTILIZER", "waste_collection"): [
        ("UTILIZER", "recycling"),        # both about waste
        ("HOUSE", "repair_request"),      # infrastructure
        ("HOUSE", "neighbor_complaint"),  # mess related
    ],
    ("UTILIZER", "recycling"): [
        ("UTILIZER", "waste_collection"), # both waste
        ("HOUSE", "utility_bill"),        # service related
        ("BOLTALKA", "chitchat"),         # general question
    ],
    # BOLTALKA domain
    ("BOLTALKA", "greeting"): [
        ("BOLTALKA", "chitchat"),         # both casual
        ("EXIT", "goodbye"),             # greeting vs farewell confusion
        ("BOLTALKA", "thanks"),           # politeness
    ],
    ("BOLTALKA", "chitchat"): [
        ("BOLTALKA", "greeting"),         # both casual
        ("BOLTALKA", "thanks"),           # general conversation
        ("EXIT", "end_conversation"),     # off-topic might end chat
    ],
    ("BOLTALKA", "thanks"): [
        ("EXIT", "goodbye"),             # politeness → closing
        ("EXIT", "end_conversation"),     # gratitude → closing
        ("BOLTALKA", "chitchat"),         # casual talk
    ],
    # PTNT domain
    ("PTNT", "patent_registration"): [
        ("PTNT", "patent_status"),        # both patent
        ("BIZ_ASSIST", "business_registration"),  # registration
        ("BIZ_ASSIST", "consultation"),   # legal advice
    ],
    ("PTNT", "patent_status"): [
        ("PTNT", "patent_registration"),  # both patent
        ("PAYMENTS", "payment_status"),   # status checking
        ("BIZ_ASSIST", "consultation"),   # inquiry
    ],
    # BIZ_ASSIST domain
    ("BIZ_ASSIST", "business_registration"): [
        ("BIZ_ASSIST", "consultation"),   # both business
        ("PTNT", "patent_registration"),  # registration
        ("BIZ_ASSIST", "subsidy"),        # government services
    ],
    ("BIZ_ASSIST", "consultation"): [
        ("BIZ_ASSIST", "business_registration"),  # both business
        ("BIZ_ASSIST", "subsidy"),        # both support
        ("PTNT", "patent_registration"),  # legal matters
    ],
    ("BIZ_ASSIST", "subsidy"): [
        ("BIZ_ASSIST", "consultation"),   # both support
        ("BIZ_ASSIST", "business_registration"),  # government
        ("PAYMENTS", "payment_status"),   # financial
    ],
    # DNEVNIK domain
    ("DNEVNIK", "grades"): [
        ("DNEVNIK", "homework"),          # both school
        ("DNEVNIK", "schedule"),          # both school
        ("BOLTALKA", "chitchat"),         # general question
    ],
    ("DNEVNIK", "homework"): [
        ("DNEVNIK", "schedule"),          # both schedule-related
        ("DNEVNIK", "grades"),            # both school
        ("BOLTALKA", "chitchat"),         # general question
    ],
    ("DNEVNIK", "schedule"): [
        ("DNEVNIK", "homework"),          # both school tasks
        ("DNEVNIK", "grades"),            # both school
        ("MOSBILET", "ticket_purchase"),  # events/schedule
    ],
    # PAYMENTS domain
    ("PAYMENTS", "payment_status"): [
        ("PAYMENTS", "payment_method"),   # both payments
        ("HOUSE", "utility_bill"),        # billing
        ("MOSBILET", "ticket_refund"),    # financial issue
    ],
    ("PAYMENTS", "payment_method"): [
        ("PAYMENTS", "payment_status"),   # both payments
        ("HOUSE", "utility_bill"),        # how to pay bills
        ("MOSBILET", "ticket_purchase"),  # purchasing
    ],
    # EXIT domain
    ("EXIT", "goodbye"): [
        ("EXIT", "end_conversation"),     # both closing
        ("BOLTALKA", "thanks"),           # polite closing
        ("BOLTALKA", "greeting"),         # confusion greeting/farewell
    ],
    ("EXIT", "end_conversation"): [
        ("EXIT", "goodbye"),             # both closing
        ("BOLTALKA", "thanks"),           # closing with gratitude
        ("BOLTALKA", "chitchat"),         # might seem off-topic
    ],
}


def classify_pair(row: dict) -> str:
    """Classify a DPO pair as clean, identical, case_only, or swapped."""
    chosen = row["chosen"]
    rejected = row["rejected"]

    if chosen == rejected:
        return "identical"

    if chosen.lower() == rejected.lower():
        return "case_only"

    try:
        c = json.loads(chosen)
        r = json.loads(rejected)
        if set(c.values()) == set(r.values()) and c != r:
            return "swapped"
    except (json.JSONDecodeError, AttributeError):
        pass

    return "clean"


def generate_hard_negative(chosen_str: str, prompt: str, used_negatives: set) -> str:
    """Generate a semantically confusable hard negative for a given chosen response."""
    try:
        chosen = json.loads(chosen_str)
    except json.JSONDecodeError:
        return chosen_str

    domain = chosen.get("domain", "")
    skill = chosen.get("skill", "")
    key = (domain, skill)

    candidates = HARD_NEGATIVES.get(key, [])
    if not candidates:
        # Fallback: pick a random different domain/skill from taxonomy
        all_options = list(HARD_NEGATIVES.keys())
        candidates = [(d, s) for d, s in all_options if d != domain or s != skill]

    # Try to pick a candidate not recently used for this prompt
    for neg_domain, neg_skill in candidates:
        cache_key = (prompt, neg_domain, neg_skill)
        if cache_key not in used_negatives:
            used_negatives.add(cache_key)
            return json.dumps({"domain": neg_domain, "skill": neg_skill}, ensure_ascii=False)

    # All used — pick first candidate anyway
    neg_domain, neg_skill = candidates[0]
    return json.dumps({"domain": neg_domain, "skill": neg_skill}, ensure_ascii=False)


def clean_dataset(input_path: str, output_path: str, backup: bool = True) -> dict:
    """Clean DPO pairs dataset.

    Args:
        input_path: Path to original JSONL file.
        output_path: Path to save cleaned JSONL file.
        backup: Whether to backup original file.

    Returns:
        Statistics dict.
    """
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Backup original
    if backup:
        backup_path = input_file.with_suffix(".jsonl.bak")
        shutil.copy2(input_file, backup_path)
        logger.info("Backup saved to %s", backup_path)

    # Read all pairs
    with open(input_file, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    logger.info("Loaded %d pairs from %s", len(rows), input_path)

    # Classify and fix
    stats = {"total": len(rows), "clean": 0, "identical_removed": 0,
             "case_only_fixed": 0, "swapped_fixed": 0}
    cleaned = []
    used_negatives: set = set()

    random.seed(42)  # Reproducibility

    for row in rows:
        category = classify_pair(row)

        if category == "identical":
            # Generate hard negative replacement
            new_rejected = generate_hard_negative(
                row["chosen"], row["prompt"], used_negatives
            )
            if new_rejected != row["chosen"]:
                row["rejected"] = new_rejected
                cleaned.append(row)
                stats["identical_removed"] += 1
            else:
                # Could not generate different negative — skip pair
                stats["identical_removed"] += 1
                logger.warning("Dropped identical pair: '%s'", row["prompt"][:50])

        elif category == "case_only":
            # Replace with hard negative
            new_rejected = generate_hard_negative(
                row["chosen"], row["prompt"], used_negatives
            )
            row["rejected"] = new_rejected
            cleaned.append(row)
            stats["case_only_fixed"] += 1

        elif category == "swapped":
            # Replace with hard negative
            new_rejected = generate_hard_negative(
                row["chosen"], row["prompt"], used_negatives
            )
            row["rejected"] = new_rejected
            cleaned.append(row)
            stats["swapped_fixed"] += 1

        else:
            cleaned.append(row)
            stats["clean"] += 1

    stats["output_total"] = len(cleaned)

    # Shuffle to avoid clustering of fixed pairs
    random.shuffle(cleaned)

    # Save cleaned dataset
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for row in cleaned:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info("Saved %d cleaned pairs to %s", len(cleaned), output_path)
    return stats


def main():
    parser = argparse.ArgumentParser(description="Clean DPO pairs dataset")
    parser.add_argument("--input", default="data/cam_dpo_pairs.jsonl",
                        help="Input JSONL file")
    parser.add_argument("--output", default="data/cam_dpo_pairs_clean.jsonl",
                        help="Output JSONL file")
    parser.add_argument("--no-backup", action="store_true",
                        help="Skip creating backup of original file")
    args = parser.parse_args()

    stats = clean_dataset(args.input, args.output, backup=not args.no_backup)

    logger.info("=" * 50)
    logger.info("Cleaning Summary:")
    logger.info("  Original pairs:     %d", stats["total"])
    logger.info("  Already clean:      %d", stats["clean"])
    logger.info("  Identical → fixed:  %d", stats["identical_removed"])
    logger.info("  Case-only → fixed:  %d", stats["case_only_fixed"])
    logger.info("  Swapped → fixed:    %d", stats["swapped_fixed"])
    logger.info("  Output pairs:       %d", stats["output_total"])
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
