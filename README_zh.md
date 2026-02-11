# 用 OpenAI SDK 学习 Claude Code

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)

[English](./README.md) | [中文](./README_zh.md)

这个仓库是优秀教程项目
[shareAI-lab/learn-claude-code](https://github.com/shareAI-lab/learn-claude-code)
的 OpenAI SDK 适配版本。

核心目标不变：从零开始，逐步理解一个 coding agent 是怎么工作的，
从最小可用的 bash agent，到支持 skills 的 agent。

---

## 这个仓库解决什么问题

如果你使用 OpenAI 兼容接口，这个仓库可以让你沿着原项目的学习路径，
直接用 OpenAI 风格的工具调用方式来实践。

你可以清晰看到每一版的能力演进：
- v0: 一个工具也能工作
- v1: 完整 agent loop + 核心工具
- v2: Todo 显式规划
- v3: Subagent 上下文隔离
- v4: Skill 按需注入知识

## 学习路径

```text
从这里开始
    |
    v
[openai_v0_bash_agent.py] ------> "一个工具就够了"
    |
    v
[openai_v1_basic_agent.py] -----> "完整 agent loop"
    |
    v
[openai_v2_todo_agent.py] ------> "让计划可见"
    |
    v
[openai_v3_subagent.py] --------> "分而治之"
    |
    v
[openai_v4_skills_agent.py] ----> "按需加载领域知识"
```

推荐顺序：
1. 先运行 `openai_v0_bash_agent.py`。
2. 对比 `openai_v0_*` 和 `openai_v1_*` 的差异。
3. 看 `openai_v2_todo_agent.py` 的规划机制。
4. 看 `openai_v3_subagent.py` 的任务拆分机制。
5. 看 `openai_v4_skills_agent.py` 的技能机制。

## 快速开始

### 1) 安装依赖

使用 `uv`（本仓库推荐）：

```bash
uv sync
```

或者使用 pip：

```bash
pip install openai python-dotenv
```

### 2) 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`：

```bash
OPENAI_API_KEY=sk-xxx
# OPENAI_BASE_URL=https://api.openai.com/v1
# MODEL_ID=gpt-4o
```

### 3) 运行各版本 agent

```bash
python openai_v0_bash_agent.py
python openai_v0_bash_agent_mini.py
python openai_v1_basic_agent.py
python openai_v2_todo_agent.py
python openai_v3_subagent.py
python openai_v4_skills_agent.py
```

## 核心模式

本教程所有 agent 的核心都是同一个循环：

```python
while True:
    response = model(messages, tools)
    if not response.tool_calls:
        return response
    tool_results = execute(response.tool_calls)
    messages.extend(tool_results)
```

其余增强都建立在这个基础之上。

## 版本对比

| 版本 | 主文件 | 核心增强 | 关键点 |
|---|---|---|---|
| v0 | `openai_v0_bash_agent.py` | 仅 bash loop | 一个工具也能完成很多事 |
| v1 | `openai_v1_basic_agent.py` | read/write/edit | Model as Agent |
| v2 | `openai_v2_todo_agent.py` | `TodoWrite` + 约束 | 显式计划让行为更稳定 |
| v3 | `openai_v3_subagent.py` | `Task` + agent type | 上下文隔离提升复杂任务表现 |
| v4 | `openai_v4_skills_agent.py` | `Skill` + loader | 不训练也能扩展能力 |

## 仓库结构

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

## 测试

本地单元测试：

```bash
python tests/test_unit.py
```

集成测试（需要 API 凭据）：

```bash
TEST_API_KEY=your_key TEST_BASE_URL=https://api.openai.com/v1 TEST_MODEL=gpt-4o python tests/test_agent.py
```

CI 配置在 `.github/workflows/test.yml`。

## 参考项目

本项目结构与方法主要参考：
- [shareAI-lab/learn-claude-code](https://github.com/shareAI-lab/learn-claude-code)

## License

Use the license configured in this repository.
