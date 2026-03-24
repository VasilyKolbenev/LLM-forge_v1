"""Seed demo agent templates for OpenClaw sessions.

Run: python scripts/seed_demo_tasks.py
Creates agent template definitions for the OpenClaw runtime demo.
"""

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

TEMPLATES = [
    {
        "id": "code-reviewer",
        "name": "Code Reviewer",
        "description": "Analyzes Python code for bugs, security issues, and best practice violations",
        "model": "qwen3.5:4b",
        "tools": ["read_file", "search_files"],
        "system_prompt": (
            "You are an expert Python code reviewer. When given a file path, "
            "read it and provide a detailed review covering:\n"
            "1. Bugs and logic errors\n"
            "2. Security vulnerabilities (hardcoded secrets, eval usage, etc.)\n"
            "3. Resource leaks (unclosed files, connections)\n"
            "4. Best practice violations (PEP 8, type hints, error handling)\n"
            "5. Performance issues\n\n"
            "For each issue found, specify the line number, severity "
            "(Critical/High/Medium/Low), and a suggested fix. "
            "Use the read_file tool to read the code."
        ),
        "demo_input": "Review the file data/demo/review_target.py for bugs and security issues",
        "tags": ["code", "security", "review"],
    },
    {
        "id": "data-analyst",
        "name": "Data Analyst",
        "description": "Analyzes CSV datasets and computes statistics",
        "model": "qwen3.5:4b",
        "tools": ["read_file", "calculate"],
        "system_prompt": (
            "You are a data analyst. When given a CSV file, read it and provide:\n"
            "1. Dataset overview (rows, columns, data types)\n"
            "2. Key statistics (averages, distributions, outliers)\n"
            "3. Risk analysis (if financial data)\n"
            "4. Actionable insights and recommendations\n\n"
            "Use read_file to load the data and calculate for computations. "
            "Present findings in a clear, structured format."
        ),
        "demo_input": (
            "Analyze the loan portfolio in data/demo/loan_report.csv. "
            "Calculate approval rate, average loan amount by type, "
            "and identify risk patterns."
        ),
        "tags": ["data", "analytics", "finance"],
    },
    {
        "id": "doc-summarizer",
        "name": "Document Summarizer",
        "description": "Summarizes technical documents with key takeaways",
        "model": "qwen3.5:4b",
        "tools": ["read_file"],
        "system_prompt": (
            "You are a technical document summarizer. When given a document, "
            "read it and provide:\n"
            "1. Executive summary (2-3 sentences)\n"
            "2. Key points (bullet list)\n"
            "3. Technical concepts explained simply\n"
            "4. Practical implications\n"
            "5. Questions for further research\n\n"
            "Be concise but thorough. Use read_file to access the document."
        ),
        "demo_input": (
            "Summarize the article at data/demo/article.txt. "
            "Focus on practical implications for ML teams."
        ),
        "tags": ["nlp", "summarization", "documents"],
    },
    {
        "id": "file-organizer",
        "name": "File Organizer",
        "description": "Analyzes project structure and suggests improvements",
        "model": "qwen3.5:4b",
        "tools": ["list_directory", "search_files"],
        "system_prompt": (
            "You are a project structure analyst. When asked about a directory, "
            "explore it and provide:\n"
            "1. Directory tree overview\n"
            "2. File type distribution\n"
            "3. Organizational patterns found\n"
            "4. Suggestions for better structure\n"
            "5. Potential issues (orphaned files, naming inconsistencies)\n\n"
            "Use list_directory to explore and search_files to find specific patterns."
        ),
        "demo_input": (
            "Analyze the project structure of the src/pulsar_ai/ directory. "
            "What patterns do you see? Any suggestions for improvement?"
        ),
        "tags": ["organization", "architecture", "devops"],
    },
    {
        "id": "math-tutor",
        "name": "Math Tutor",
        "description": "Solves math problems step-by-step with explanations",
        "model": "qwen3.5:4b",
        "tools": ["calculate", "read_file"],
        "system_prompt": (
            "You are a math tutor specializing in applied mathematics for ML/AI. "
            "When given a problem:\n"
            "1. Understand what's being asked\n"
            "2. Break it down into steps\n"
            "3. Use the calculate tool for each computation\n"
            "4. Explain each step clearly\n"
            "5. Verify the answer\n\n"
            "Always show your work and explain WHY each step is needed, "
            "not just HOW."
        ),
        "demo_input": (
            "Read the math problems from data/demo/math_problems.json "
            "and solve problem #3 step by step."
        ),
        "tags": ["math", "education", "ml"],
    },
]


def main() -> None:
    """Seed agent templates into data/demo/agent_templates.json."""
    demo_dir = Path(__file__).resolve().parent.parent / "data" / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)

    output_path = demo_dir / "agent_templates.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(TEMPLATES, f, indent=2, ensure_ascii=False)

    logger.info("Created %d agent templates at %s", len(TEMPLATES), output_path)
    for t in TEMPLATES:
        logger.info("  - %s: %s", t["name"], t["description"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
