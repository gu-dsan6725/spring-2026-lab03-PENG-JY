# Comparison: Claude Code vs. Google Antigravity

This document compares the features and workflows of **Claude Code** (Part 1) and **Google Antigravity** (Part 2) based on our experience building the Wine Classification pipeline.

## Feature Comparison Table

| Concept | Claude Code | Google Antigravity |
|---|---|---|
| **Project instructions** | `CLAUDE.md` (Single file) | `GEMINI.md` + `.agent/rules/*.md` (Modular files) |
| **Reusable AI capabilities** | Skills (`.claude/skills/*.md`) | Skills (`.agent/skills/*/SKILL.md`) |
| **On-demand commands** | Slash commands (`.claude/commands/*.md`) | Workflows (`.agent/workflows/*.md`) |
| **Automated checks** | Native Hooks (`PreToolUse`, `PostToolUse`) | No native equivalent (Rely on `pre-commit` framework) |
| **Task decomposition** | Subagents (Automatic via Task tool) | Manager View (Manual orchestration of multiple agents) |
| **Execution control** | Plan mode (`/plan`) | Terminal execution policies (Off, Auto, Turbo) |

## Detailed Reflections

### 1. Project Instructions & Rules
- **Claude Code**: Uses a single `CLAUDE.md` file. It's simple and centralized but can get crowded for large projects.
- **Antigravity**: Splits instructions into `GEMINI.md` (global/high-level) and `.agent/rules/` (specific topics like `code-style-guide.md`). This modularity is better for scaling but requires managing multiple files.

### 2. Automated Quality Checks
- **Claude Code**: Has powerful **native hooks**. We saw this when `ruff` ran automatically after every file edit without us configuring git hooks.
- **Antigravity**: Relies on external tools like `pre-commit`. While flexible, it implies the agent doesn't "know" about the checks until they fail at commit time, or unless explicitly instructed in a Workflow.

### 3. Task Decomposition
- **Claude Code**: Spawns **Subagents** automatically to handle sub-tasks (e.g., "Research", "Coding"). It feels more fluid and autonomous.
- **Antigravity**: Uses a **Manager View** where the user explicitly creates and assigns tasks to parallel agents. This offers more control and visibility but requires more user management.

### 4. Workflow & Control
- **Claude Code**: Deeply integrated `/plan` mode acts as a scratchpad and guide.
- **Antigravity**: **Workflows** are essentially saved prompts that guide the agent step-by-step. The **Execution Policy** (Auto/Turbo) gives the user fine-grained control over when the agent can run terminal commands.

## Conclusion

- **Claude Code** feels more like an autonomous pair programmer that handles the "meta" work (quality checks, sub-tasking) automatically.
- **Antigravity** feels like a powerful, modular platform that gives the user more structural control (rules, workflows, multiple agents) but expects the user to set up the infrastructure (pre-commit, manual orchestration).
