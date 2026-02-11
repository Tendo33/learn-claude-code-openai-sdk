#!/usr/bin/env python
"""v0_bash_agent_mini.py - Mini GPT Code (Compact) - OpenAI SDK Version"""

import json
import os
import subprocess as sp
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)
C = OpenAI(base_url=os.getenv("OPENAI_BASE_URL"))
M = os.getenv("MODEL_ID", "gpt-4o")
T = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Shell cmd. Read:cat/grep/find/rg/ls. Write:echo>/sed. Subagent(for complex subtask): python openai_v0_bash_agent_mini.py 'task'",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }
]
S = f"CLI agent at {os.getcwd()}. Use bash to solve problems. Spawn subagent for complex subtasks: python openai_v0_bash_agent_mini.py 'task'. Subagent isolates context and returns summary. Be concise."


def chat(p, h=[]):
    h.append({"role": "user", "content": p})
    while True:
        r = C.chat.completions.create(
            model=M,
            messages=[{"role": "system", "content": S}] + h,
            tools=T,
            max_tokens=8000,
        )
        m = r.choices[0].message
        if not m.tool_calls:
            h.append({"role": "assistant", "content": m.content or ""})
            return m.content or ""
        am = {"role": "assistant", "content": m.content}
        am["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in m.tool_calls
        ]
        h.append(am)
        for tc in m.tool_calls:
            a = json.loads(tc.function.arguments)
            print(f"\033[33m$ {a['command']}\033[0m")
            o = sp.run(a["command"], shell=1, capture_output=1, text=1, timeout=300)
            print(o.stdout + o.stderr or "(empty)")
            h.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": (o.stdout + o.stderr)[:50000],
                }
            )


if __name__ == "__main__":
    [print(chat(sys.argv[1]))] if len(sys.argv) > 1 else [
        print(chat(q, h))
        for h in [[]]
        for _ in iter(int, 1)
        if (q := input("\033[36m>> \033[0m")) not in ("q", "")
    ]
