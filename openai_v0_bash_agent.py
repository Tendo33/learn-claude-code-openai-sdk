#!/usr/bin/env python
"""
v0_bash_agent.py - Mini GPT Code: Bash is All You Need (~50 lines core) - OpenAI SDK Version

Core Philosophy: "Bash is All You Need"
======================================
This is the ULTIMATE simplification of a coding agent. After building v1-v3,
we ask: what is the ESSENCE of an agent?

The answer: ONE tool (bash) + ONE loop = FULL agent capability.

Why Bash is Enough:
------------------
Unix philosophy says everything is a file, everything can be piped.
Bash is the gateway to this world:

    | You need      | Bash command                           |
    |---------------|----------------------------------------|
    | Read files    | cat, head, tail, grep                  |
    | Write files   | echo '...' > file, cat << 'EOF' > file |
    | Search        | find, grep, rg, ls                     |
    | Execute       | python, npm, make, any command         |
    | **Subagent**  | python openai_v0_bash_agent.py "task"  |

The last line is the KEY INSIGHT: calling itself via bash implements subagents!
No Task tool, no Agent Registry - just recursion through process spawning.

How Subagents Work:
------------------
    Main Agent
      |-- bash: python openai_v0_bash_agent.py "analyze architecture"
           |-- Subagent (isolated process, fresh history)
                |-- bash: find . -name "*.py"
                |-- bash: cat src/main.py
                |-- Returns summary via stdout

Process isolation = Context isolation:
- Child process has its own history=[]
- Parent captures stdout as tool result
- Recursive calls enable unlimited nesting

Usage:
    # Interactive mode
    python openai_v0_bash_agent.py

    # Subagent mode (called by parent agent or directly)
    python openai_v0_bash_agent.py "explore src/ and summarize"

OpenAI SDK Key Differences from Anthropic SDK:
----------------------------------------------
- Tool definitions use {"type": "function", "function": {"name", "description", "parameters"}}
- System prompt goes into messages as {"role": "system", ...}, not a separate parameter
- Response: response.choices[0].message (not response.content)
- Tool calls: message.tool_calls[i].function.name/.arguments (JSON string, needs json.loads)
- Stop reason: finish_reason == "tool_calls" (not "tool_use")
- Tool results: separate {"role": "tool"} messages (not bundled in user message)
"""

import json
import os
import subprocess
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

# Initialize OpenAI client (uses OPENAI_API_KEY and OPENAI_BASE_URL env vars)
client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL"))
MODEL = os.getenv("MODEL_ID", "gpt-4o")

# The ONE tool that does everything
# OpenAI format: wrapped in {"type": "function", "function": {...}}
# Uses "parameters" instead of Anthropic's "input_schema"
TOOL = [{
    "type": "function",
    "function": {
        "name": "bash",
        "description": """Execute shell command. Common patterns:
- Read: cat/head/tail, grep/find/rg/ls, wc -l
- Write: echo 'content' > file, sed -i 's/old/new/g' file
- Subagent: python openai_v0_bash_agent.py 'task description' (spawns isolated agent, returns summary)""",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"]
        }
    }
}]

# System prompt teaches the model HOW to use bash effectively
SYSTEM = f"""You are a CLI agent at {os.getcwd()}. Solve problems using bash commands.

Rules:
- Prefer tools over prose. Act first, explain briefly after.
- Read files: cat, grep, find, rg, ls, head, tail
- Write files: echo '...' > file, sed -i, or cat << 'EOF' > file
- Subagent: For complex subtasks, spawn a subagent to keep context clean:
  python openai_v0_bash_agent.py "explore src/ and summarize the architecture"

When to use subagent:
- Task requires reading many files (isolate the exploration)
- Task is independent and self-contained
- You want to avoid polluting current conversation with intermediate details

The subagent runs in isolation and returns only its final summary."""


def chat(prompt, history=None):
    """
    The complete agent loop in ONE function.

    This is the core pattern that ALL coding agents share:
        while not done:
            response = model(messages, tools)
            if no tool calls: return
            execute tools, append results

    OpenAI SDK differences:
    - System prompt is the first message in the messages list
    - Tool call arguments come as JSON strings (need json.loads)
    - Each tool result is a separate message with role="tool"
    - finish_reason is "tool_calls" (not "tool_use")

    Args:
        prompt: User's request
        history: Conversation history (mutable, shared across calls in interactive mode)

    Returns:
        Final text response from the model
    """
    if history is None:
        history = []

    history.append({"role": "user", "content": prompt})

    while True:
        # 1. Call the model with tools
        # OpenAI: system prompt goes in messages, not as separate param
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM}] + history,
            tools=TOOL,
            max_tokens=8000
        )

        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        # 2. If model didn't call tools, we're done
        # OpenAI: finish_reason is "stop" when done, "tool_calls" when calling tools
        if finish_reason != "tool_calls":
            history.append({"role": "assistant", "content": message.content or ""})
            return message.content or ""

        # 3. Build assistant message for history
        # OpenAI: assistant message needs "tool_calls" field with serialized tool calls
        assistant_msg = {"role": "assistant", "content": message.content}
        assistant_msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                }
            }
            for tc in message.tool_calls
        ]
        history.append(assistant_msg)

        # 4. Execute each tool call and collect results
        # OpenAI: each tool result is a SEPARATE message with role="tool"
        # (Anthropic bundles all results in one "user" message)
        for tc in message.tool_calls:
            # OpenAI: arguments are JSON strings, need json.loads
            args = json.loads(tc.function.arguments)
            cmd = args["command"]
            print(f"\033[33m$ {cmd}\033[0m")  # Yellow color for commands

            try:
                out = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=300,
                    cwd=os.getcwd()
                )
                output = out.stdout + out.stderr
            except subprocess.TimeoutExpired:
                output = "(timeout after 300s)"

            print(output or "(empty)")

            # 5. Append each result as separate "tool" message
            history.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": output[:50000]  # Truncate very long outputs
            })


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Subagent mode: execute task and print result
        print(chat(sys.argv[1]))
    else:
        # Interactive REPL mode
        history = []
        while True:
            try:
                query = input("\033[36m>> \033[0m")  # Cyan prompt
            except (EOFError, KeyboardInterrupt):
                break
            if query in ("q", "exit", ""):
                break
            print(chat(query, history))
