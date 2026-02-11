# Learn Claude Code with OpenAI SDK

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)

[English](./README.md) | [中文](./README_zh.md)

This repository is an OpenAI SDK adaptation of the excellent educational project
[shareAI-lab/learn-claude-code](https://github.com/shareAI-lab/learn-claude-code).

The goal is the same: learn how coding agents work by building them step by step,
from a minimal bash agent to a skills-enabled agent.

---

## Why this repo

If you prefer OpenAI-compatible APIs, this repo gives you the same learning path
with OpenAI tool-calling patterns and environment variables.

You can compare versions side by side and understand what each upgrade adds:
- v0: one tool is enough
- v1: full agent loop with core tools
- v2: explicit todo planning
- v3: subagents with context isolation
- v4: skill loading for domain knowledge

## Learning path

```text
Start here
    |
    v
[openai_v0_bash_agent.py] ------> "One tool is enough"
    |
    v
[openai_v1_basic_agent.py] -----> "The complete agent loop"
    |
    v
[openai_v2_todo_agent.py] ------> "Make plans explicit"
    |
    v
[openai_v3_subagent.py] --------> "Divide and conquer"
    |
    v
[openai_v4_skills_agent.py] ----> "Knowledge on demand"
```

Recommended order:
1. Run `openai_v0_bash_agent.py` first.
2. Compare `openai_v0_*` and `openai_v1_*`.
3. Study `openai_v2_todo_agent.py` for planning behavior.
4. Study `openai_v3_subagent.py` for task decomposition.
5. Study `openai_v4_skills_agent.py` for the skills mechanism.

## Quick start

### 1) Install dependencies

Using `uv` (recommended in this repo):

```bash
uv sync
```

Or with pip:

```bash
pip install openai python-dotenv
```

Optional (enable Logfire):

```bash
uv sync --extra logfire
# or
pip install ".[logfire]"
```

### 2) Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```bash
OPENAI_API_KEY=sk-xxx
# OPENAI_BASE_URL=https://api.openai.com/v1
# MODEL_ID=gpt-4o
# ENABLE_LOGFIRE=true
# LOGFIRE_TOKEN=your-logfire-write-token
```

### 3) Run agents

```bash
python openai_v0_bash_agent.py
python openai_v0_bash_agent_mini.py
python openai_v1_basic_agent.py
python openai_v2_todo_agent.py
python openai_v3_subagent.py
python openai_v4_skills_agent.py
```

## Core pattern

Every coding agent in this tutorial follows the same loop:

```python
while True:
    response = model(messages, tools)
    if not response.tool_calls:
        return response
    tool_results = execute(response.tool_calls)
    messages.extend(tool_results)
```

Everything else is an iteration on this pattern.

## Version comparison

| Version | Main file | Core addition | Key idea |
|---|---|---|---|
| v0 | `openai_v0_bash_agent.py` | bash-only loop | One tool can go very far |
| v1 | `openai_v1_basic_agent.py` | read/write/edit tools | Model as agent |
| v2 | `openai_v2_todo_agent.py` | `TodoWrite` + constraints | Explicit plans improve reliability |
| v3 | `openai_v3_subagent.py` | `Task` tool + agent types | Context isolation for complex work |
| v4 | `openai_v4_skills_agent.py` | `Skill` tool + loader | Add expertise without retraining |

## Repository structure

```text
learn-claude-code-openai-sdk/
├── openai_v0_bash_agent.py
├── openai_v0_bash_agent_mini.py
├── openai_v1_basic_agent.py
├── openai_v2_todo_agent.py
├── openai_v3_subagent.py
├── openai_v4_skills_agent.py
├── skills/
├── tests/
│   ├── test_unit.py
│   └── test_agent.py
├── .github/workflows/test.yml
├── .env.example
└── pyproject.toml
```

## Tests

Run locally:

```bash
python tests/test_unit.py
```

Integration tests require API credentials:

```bash
TEST_API_KEY=your_key TEST_BASE_URL=https://api.openai.com/v1 TEST_MODEL=gpt-4o python tests/test_agent.py
```

CI workflow is at `.github/workflows/test.yml`.

## Reference

This project references and follows the structure/philosophy of:
- [shareAI-lab/learn-claude-code](https://github.com/shareAI-lab/learn-claude-code)

## License

[MIT License](./LICENSE)
