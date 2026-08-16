---
name: advanced-agentic-workflows
description: Enforces next-generation agent workflows (Spec-driven development, AST maps, reusable skills, and automated hooks) for all interactions in this project.
trigger: always_on
---

# Advanced Agentic Workflows & Paradigms

As an autonomous agent operating in this repository, you must permanently adopt the following advanced paradigms over traditional flat-file LLM coding:

## 1. Spec-Driven Development (Living Source of Truth)
Do not treat the codebase as the sole source of truth. The **Implementation Plan** (`implementation_plan.md` and `task.md`) is the absolute living spec. Code is merely the downstream output generated (and regenerated) against this spec. Always update the spec *before* refactoring code.

## 2. Dependency & AST Graphs > Flat Navigation
Instead of flat `ls` and `cat` commands, navigate codebases via structural and semantic boundaries. Use Python's `ast` module or recursive dependency mapping to understand how a change in `DataEngine` cascades into `DeskVerdict`. Understand the *repo map*, not just the file list.

## 3. Shareable Skill Marketplaces
Stop solving the same problem twice. When you derive a complex, repeatable procedure, package it immediately into a standalone `SKILL.md`. Treat procedures as reusable, shareable assets.

## 4. Hooks Enforcing Checks (System > Memory)
Never rely on the agent's "memory" to run tests or formatting checks. If a verification is required, build it into a hook (e.g., `pre-commit`, Pytest fixtures, or Antigravity's `hooks.json`). The harness must enforce the invariants automatically so the agent is freed to focus purely on reasoning.

## 5. Agent Workflow Evaluation
Do not just evaluate the generated code. Evaluate the *workflow*. Build test harnesses (like `verify_all_modules.py`) that specifically audit if the multi-agent pipeline is correctly gating decisions, enforcing limits, and communicating properly.
