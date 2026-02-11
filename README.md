# Learn Claude Code with OpenAI SDK

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

[English](./README.md) | [中文](./README_zh.md)

This repository extends [shareAI-lab/learn-claude-code](https://github.com/shareAI-lab/learn-claude-code) with OpenAI SDK-specific implementations.

## What is added in this repo

- OpenAI SDK versions for each tutorial stage:
  - `openai_v0_bash_agent.py`
  - `openai_v1_basic_agent.py`
  - `openai_v2_todo_agent.py`
  - `openai_v3_subagent.py`
  - `openai_v4_skills_agent.py`
- OpenAI-compatible environment variable setup (`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `MODEL_ID`).
- Optional Logfire integration via extra dependency (`.[logfire]`), providing clear interaction records with the model.
- Test scripts and CI workflow adapted for this OpenAI SDK version.

## Quick start

```bash
cp .env.example .env
pip install -e ".[logfire]"
# or without logfire:
# pip install -e .
```
[logfire](https://logfire.pydantic.dev/docs/) doc

Run examples:

```bash
python openai_v0_bash_agent.py
python openai_v1_basic_agent.py
python openai_v2_todo_agent.py
python openai_v3_subagent.py
python openai_v4_skills_agent.py
```

## Tests

```bash
python tests/test_unit.py
```

Integration tests:

```bash
TEST_API_KEY=your_key TEST_BASE_URL=https://api.openai.com/v1 TEST_MODEL=gpt-4o python tests/test_agent.py
```

## License

[MIT License](./LICENSE)
