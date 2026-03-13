# AGENTS.md

## Mission

This repository should be advanced in a phased, market-oriented way. The priority is to move the product from demo-ready toward production-capable without destabilizing the working base.

## Required Reading Before Work

Before starting any meaningful task, read these files:

1. `docs/CLAUDE_CODE_ROADMAP.md`
2. `docs/CLAUDE_CODE_EXECUTION_PROMPT.md`
3. `docs/MULTI_AGENT_OPERATING_MODEL.md`

If the task touches architecture, persistence, auth, compute, protocols, or release readiness, those files are mandatory context.

## Operating Rules

1. Work one major phase at a time unless the user explicitly requests cross-phase work.
2. Default phase priority:
   - Phase 1: persistence layer
   - Phase 2: durable jobs and sessions
   - Phase 3: security hardening
   - Phase 4: unified pipeline execution
   - Phase 5: remote compute hardening
   - Phase 6: frontend decomposition and tests
   - Phase 7: protocols and assistant productionization
   - Phase 8: test harness and CI stabilization
3. Read the relevant implementation files and tests before editing code.
4. Prefer vertical slices that end in a verifiable result over large unfinished refactors.
5. After every substantial change, run the most relevant tests, build, or smoke check available.
6. If a risky migration is involved, minimize blast radius and preserve compatibility where practical.
7. When a new important debt is discovered, document it in the roadmap or in a new docs file.

## Multi-Agent Workflow

When more than one coding agent is used on this repository:

1. Assign a single owner for each task packet.
2. Do not have two agents edit the same file set at the same time unless the user explicitly wants parallel exploration.
3. Prefer separate branches or worktrees for parallel experiments.
4. Use one agent for implementation and another for review, testing, risk scanning, or follow-up fixes.
5. Record handoff notes in the final response or in a task-specific docs file when the work spans multiple sessions.

## What Good Progress Looks Like

A good iteration should produce:

- concrete code or documentation progress;
- clear verification steps;
- a short list of residual risks;
- one obvious next step.

Avoid long analysis without implementation when implementation is feasible.

## Market-Readiness Bias

When choosing between tasks, prioritize work that improves one of these areas:

1. reliability and persistence;
2. security and secret handling;
3. execution consistency;
4. testability and CI signal;
5. maintainability of core product surfaces;
6. operational visibility and release readiness.

## Delivery Format

At the end of each task, report:

1. what changed;
2. what was verified;
3. what risks remain;
4. the next best step.
