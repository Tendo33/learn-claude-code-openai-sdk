# 用 OpenAI SDK 学习 Claude Code

[English](./README.md) | [中文](./README_zh.md)

本仓库是 [shareAI-lab/learn-claude-code](https://github.com/shareAI-lab/learn-claude-code) 项目的 **OpenAI SDK 移植版**。它提供了使用 OpenAI 的 Chat Completions 和 Tool-call API 实现的 12 阶段 Agent 治理机制演进过程。

## 项目结构

- **`agents/`**: 包含从 `s01` 到 `s12` 的 12 个演进阶段。
- **`agents/s_full.py`**: 完整的集成参考实现。
- **`web/`**: 用于可视化 Agent 演进的 Next.js 学习门户。
- **`docs/`**: 各阶段的多语言文档。

## 快速开始

1. **环境配置**:
   ```bash
   cp .env.example .env
   # 填入 OPENAI_API_KEY 和 OPENAI_BASE_URL
   ```

2. **安装依赖**:
   ```bash
   pip install -e .
   ```

3. **运行 Agent**:
   ```bash
   python agents/s_full.py
   ```

## 已移植的核心机制

- **s01-s05**: 核心循环、工具调用、规划、子代理、技能加载。
- **s06**: 上下文三层压缩机制。
- **s07-s08**: 持久化任务看板与后台线程执行。
- **s09-s11**: 代理团队协作、握手协议、自主认领任务。
- **s12**: 工作树（Worktree）级别的任务隔离。

## 许可证

[MIT License](./LICENSE)
