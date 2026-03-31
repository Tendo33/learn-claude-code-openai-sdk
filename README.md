# Learn Claude Code with OpenAI SDK

[English](./README.md) | [中文](./README_zh.md)

This repository is an **OpenAI SDK port** of the [shareAI-lab/learn-claude-code](https://github.com/shareAI-lab/learn-claude-code) project. It provides a 12-stage evolution of agent harness mechanisms implemented using OpenAI's chat completions and tool-call APIs.

## Project Structure

- **`agents/`**: 12 stages of agent evolution (`s01` to `s12`).
- **`agents/s_full.py`**: The integrated reference implementation.
- **`web/`**: Learning portal (Next.js) for visualizing agents.
- **`docs/`**: Multilingual documentation for each stage.

## Quick Start

1. **Configure Environment**:
   ```bash
   cp .env.example .env
   # Set OPENAI_API_KEY and OPENAI_BASE_URL
   ```

2. **Install Dependencies**:
   ```bash
   pip install -e .
   ```

3. **Run Agent**:
   ```bash
   python agents/s_full.py
   ```

## Key Mechanisms Ported

- **s01-s05**: Core loop, Tool use, Planning, Subagents, Skills.
- **s06**: Context compression (micro/auto/manual).
- **s07-s08**: Persistent tasks and background threaded execution.
- **s09-s11**: Multi-agent teams, handshakes, and autonomous task claiming.
- **s12**: Worktree-level task isolation.

## License

[MIT License](./LICENSE)
