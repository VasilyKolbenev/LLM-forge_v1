# Multi-Agent Operating Model

## Purpose

This document defines how to run Codex and Claude Code together on the same product without creating chaos. The goal is not just to ship code, but to ship code that is progressively more ready for real users and eventual market release.

Use this document together with:

- `docs/CLAUDE_CODE_ROADMAP.md`
- `docs/CLAUDE_CODE_EXECUTION_PROMPT.md`
- `AGENTS.md`

## Core Model

Use a two-agent pattern:

- Claude Code acts as the phase owner and primary delivery agent.
- Codex acts as the parallel reviewer, risk scanner, test runner, docs maintainer, or secondary implementer on isolated tasks.

This creates leverage without letting both agents fight over the same context.

## Default Role Split

### Claude Code

Best used for:

- taking a roadmap phase and driving it forward end to end;
- implementing the main vertical slice;
- making coordinated multi-file changes;
- reconciling product intent, architecture, and delivery.

### Codex

Best used for:

- reviewing diffs and finding regressions;
- writing or extending tests;
- performing focused refactors after the main direction is chosen;
- scanning for market-readiness gaps;
- running recurring automation loops.

## Golden Rule

Do not let both agents actively edit the same set of files in the same branch at the same time.

Preferred options:

1. different branches;
2. different worktrees;
3. sequential handoff where one agent finishes and the other reviews.

## Recommended Delivery Loop

### Loop A. Primary build loop

1. Claude Code reads `docs/CLAUDE_CODE_ROADMAP.md` and `docs/CLAUDE_CODE_EXECUTION_PROMPT.md`.
2. Claude Code takes one phase or subtask.
3. Claude Code implements a small but complete vertical slice.
4. Claude Code runs relevant verification.
5. Claude Code reports what changed, what was verified, and what remains.

### Loop B. Secondary review loop

1. Codex reviews the resulting diff or repository state.
2. Codex looks for regressions, missing tests, security issues, operational gaps, and release blockers.
3. Codex either:
   - produces a review summary; or
   - takes a small isolated follow-up task in a separate branch or worktree.

### Loop C. Release readiness loop

1. Codex or Claude Code audits the project against the roadmap.
2. Update the backlog based on real blockers.
3. Convert findings into the next task packet.

## Task Packet Format

Every meaningful task should be framed like this:

### Goal

One concrete outcome.

### Scope

The smallest file and module surface that can solve it.

### Constraints

Compatibility, migration, security, or time limits.

### Verification

The exact tests, build commands, or smoke checks to run.

### Done When

A short definition of done with observable conditions.

## Recommended Task Types

Use Claude Code for:

- persistence migrations;
- auth and security rewrites;
- execution-path unification;
- durable runtime state work;
- larger frontend decomposition tasks.

Use Codex for:

- regression scans after each merge-sized change;
- daily roadmap progress checks;
- flaky test discovery;
- bundle-size and performance checks;
- release-blocker summaries;
- targeted test creation.

## Handoff Template

When one agent hands work to the other, include:

### Completed

- what is already changed;
- what commands were run;
- what passed and failed.

### Open Risks

- migration risks;
- edge cases;
- missing tests;
- operational concerns.

### Next Best Step

- the one task the receiving agent should do next.

## Market-Readiness Checklist

Use this checklist repeatedly during development.

### Product reliability

- durable persistence for critical state;
- restart-safe runtime behavior;
- clear terminal states for long-running tasks.

### Security

- no secrets in logs;
- no secrets in URL parameters;
- explicit auth behavior;
- auditable key handling.

### Operability

- actionable logs;
- traceable run states;
- recoverable stale jobs;
- observable remote execution failures.

### Delivery confidence

- relevant tests exist;
- CI or local checks give a trustworthy signal;
- docs reflect current behavior.

### UX and maintainability

- large screens are decomposed;
- major flows are testable;
- warnings and silent failures are reduced.

## Suggested Human Workflow

Use this as the practical day-to-day operating model:

1. Start Claude Code on the next roadmap phase.
2. Let Claude Code finish one meaningful slice.
3. Ask Codex to review the result or attack a narrow adjacent task.
4. Merge or reconcile changes.
5. Update roadmap only when a real new fact is learned.
6. Repeat.

## Minimal Manual Prompts

### Prompt for Claude Code

```md
Read `AGENTS.md`, `docs/CLAUDE_CODE_ROADMAP.md`, `docs/CLAUDE_CODE_EXECUTION_PROMPT.md`, and `docs/MULTI_AGENT_OPERATING_MODEL.md`.

Take the next highest-priority task in the current roadmap phase, implement a small but complete vertical slice, run the relevant checks, and report:
- what changed;
- what was verified;
- what risks remain;
- the next best step.
```

### Prompt for Codex

```md
Read `AGENTS.md` and `docs/MULTI_AGENT_OPERATING_MODEL.md`.

Review the current repository state against `docs/CLAUDE_CODE_ROADMAP.md`. Identify the highest-signal regressions, market-readiness gaps, missing tests, or security risks. If a narrow follow-up fix is obvious and isolated, implement it; otherwise produce a concise review with the next best task.
```

## When to Automate

Automate repeated loops, not ambiguous product design.

Good automation candidates:

- daily roadmap drift check;
- daily regression and test gap scan;
- recurring release-readiness summary;
- stale run and reliability audit;
- frontend bundle and warning scan.

Keep major architectural changes human-triggered unless the task packet is already very explicit.
