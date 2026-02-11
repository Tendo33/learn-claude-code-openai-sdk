# 用 OpenAI SDK 学习 Claude Code

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

[English](./README.md) | [中文](./README_zh.md)

本仓库是在 `shareAI-lab/learn-claude-code` 基础上，补充 OpenAI SDK 版本实现。

## 本仓库新增内容

- 对应教程各阶段的 OpenAI SDK 脚本：
  - `openai_v0_bash_agent.py`
  - `openai_v1_basic_agent.py`
  - `openai_v2_todo_agent.py`
  - `openai_v3_subagent.py`
  - `openai_v4_skills_agent.py`
- 面向 OpenAI 兼容接口的环境变量配置（`OPENAI_API_KEY`、`OPENAI_BASE_URL`、`MODEL_ID`）。
- 可选 Logfire 集成（通过 `.[logfire]` 扩展依赖），提供清晰的每次与模型的交互记录。
- 适配本仓库实现的测试脚本与 CI 工作流。

## 快速开始

```bash
cp .env.example .env
pip install -e ".[logfire]"
# 不使用 logfire:
# pip install -e .
```

运行示例：

```bash
python openai_v0_bash_agent.py
python openai_v1_basic_agent.py
python openai_v2_todo_agent.py
python openai_v3_subagent.py
python openai_v4_skills_agent.py
```

## 测试

```bash
python tests/test_unit.py
```

集成测试：

```bash
TEST_API_KEY=your_key TEST_BASE_URL=https://api.openai.com/v1 TEST_MODEL=gpt-4o python tests/test_agent.py
```

## 许可证

[MIT License](./LICENSE)
