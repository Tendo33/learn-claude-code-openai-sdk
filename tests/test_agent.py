"""
Integration tests for learn-claude-code-openai-sdk agents.

Comprehensive agent task tests covering v0-v4 core capabilities.
Runs on GitHub Actions (Linux).
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_client():
    """Get OpenAI-compatible client for testing."""
    from openai import OpenAI

    api_key = os.getenv("TEST_API_KEY")
    base_url = os.getenv("TEST_BASE_URL")
    if not api_key:
        return None
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


MODEL = os.getenv("TEST_MODEL", "gpt-4o")


# =============================================================================
# Tool Definitions
# =============================================================================

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a shell command",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}

READ_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read contents of a file",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
}

WRITE_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Write content to a file (creates or overwrites)",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
}

EDIT_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "edit_file",
        "description": "Replace old_text with new_text in a file",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            },
            "required": ["path", "old_text", "new_text"],
        },
    },
}

TODO_WRITE_TOOL = {
    "type": "function",
    "function": {
        "name": "TodoWrite",
        "description": "Update the todo list to track task progress",
        "parameters": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                            },
                            "activeForm": {"type": "string"},
                        },
                        "required": ["content", "status", "activeForm"],
                    },
                }
            },
            "required": ["items"],
        },
    },
}

V1_TOOLS = [BASH_TOOL, READ_FILE_TOOL, WRITE_FILE_TOOL, EDIT_FILE_TOOL]
V2_TOOLS = V1_TOOLS + [TODO_WRITE_TOOL]


# =============================================================================
# Agent Loop Runner
# =============================================================================

def execute_tool(name, args, workdir):
    """Execute a tool and return output."""
    import subprocess

    if name == "bash":
        cmd = args.get("command", "")
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=30, cwd=workdir
            )
            return result.stdout + result.stderr or "(empty)"
        except Exception as e:
            return f"Error: {e}"

    if name == "read_file":
        path = args.get("path", "")
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"Error: {e}"

    if name == "write_file":
        path = args.get("path", "")
        content = args.get("content", "")
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Written {len(content)} bytes to {path}"
        except Exception as e:
            return f"Error: {e}"

    if name == "edit_file":
        path = args.get("path", "")
        old = args.get("old_text", "")
        new = args.get("new_text", "")
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            if old not in content:
                return f"Error: '{old}' not found in file"
            content = content.replace(old, new, 1)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Replaced in {path}"
        except Exception as e:
            return f"Error: {e}"

    if name == "TodoWrite":
        items = args.get("items", [])
        result = []
        for item in items:
            status_icon = {
                "pending": "[ ]",
                "in_progress": "[>]",
                "completed": "[x]",
            }.get(item["status"], "[ ]")
            result.append(f"{status_icon} {item['content']}")
        completed = len([i for i in items if i["status"] == "completed"])
        return "\n".join(result) + f"\n({completed}/{len(items)} completed)"

    return f"Unknown tool: {name}"


def run_agent_loop(client, task, tools, workdir=None, max_turns=15, system_prompt=None):
    """
    Run a complete agent loop until done or max_turns.
    Returns (final_response, tool_calls_made, messages)
    """
    if workdir is None:
        workdir = os.getcwd()

    if system_prompt is None:
        system_prompt = (
            f"You are a coding agent at {workdir}. Use tools to complete tasks. Be concise."
        )

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": task}]

    tool_calls_made = []

    for _ in range(max_turns):
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=tools, max_tokens=1500
        )

        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        if finish_reason == "stop" or not message.tool_calls:
            return message.content, tool_calls_made, messages

        messages.append(
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ],
            }
        )

        for tool_call in message.tool_calls:
            func_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            tool_calls_made.append((func_name, args))

            output = execute_tool(func_name, args, workdir)
            messages.append(
                {"role": "tool", "tool_call_id": tool_call.id, "content": output[:5000]}
            )

    return None, tool_calls_made, messages


# =============================================================================
# v0 Tests: Bash Only
# =============================================================================

def test_v0_bash_echo():
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    response, calls, _ = run_agent_loop(
        client,
        "Run 'echo hello world' and tell me the output.",
        [BASH_TOOL],
    )

    assert len(calls) >= 1, "Should make at least 1 tool call"
    assert any("echo" in str(c) for c in calls), "Should run echo"
    assert response and "hello" in response.lower()

    print(f"Tool calls: {len(calls)}")
    print("PASS: test_v0_bash_echo")
    return True


def test_v0_bash_pipeline():
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "data.txt"), "w", encoding="utf-8") as f:
            f.write("apple\nbanana\napricot\ncherry\n")

        response, calls, _ = run_agent_loop(
            client,
            f"Count how many lines in {tmpdir}/data.txt start with 'a'. Use grep and wc.",
            [BASH_TOOL],
            workdir=tmpdir,
        )

        assert len(calls) >= 1
        assert response and "2" in response

    print(f"Tool calls: {len(calls)}")
    print("PASS: test_v0_bash_pipeline")
    return True


# =============================================================================
# v1 Tests: 4 Core Tools
# =============================================================================

def test_v1_read_file():
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "secret.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("The secret code is: XYZ123")

        response, calls, _ = run_agent_loop(
            client,
            f"Read {filepath} and tell me what the secret code is.",
            V1_TOOLS,
            workdir=tmpdir,
        )

        assert any(c[0] == "read_file" for c in calls), "Should use read_file"
        assert response and "XYZ123" in response

    print(f"Tool calls: {len(calls)}")
    print("PASS: test_v1_read_file")
    return True


def test_v1_write_file():
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "greeting.txt")
        response, calls, _ = run_agent_loop(
            client,
            f"Create a file at {filepath} containing 'Hello, Agent!' using write_file tool.",
            V1_TOOLS,
            workdir=tmpdir,
        )

        assert any(c[0] == "write_file" for c in calls), "Should use write_file"
        assert os.path.exists(filepath)
        with open(filepath, encoding="utf-8") as f:
            content = f.read()
        assert "Hello" in content

    print(f"Tool calls: {len(calls)}")
    print("PASS: test_v1_write_file")
    return True


def test_v1_edit_file():
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "config.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("debug=false\nport=8080\n")

        response, calls, _ = run_agent_loop(
            client,
            f"Edit {filepath} to change debug=false to debug=true using edit_file tool.",
            V1_TOOLS,
            workdir=tmpdir,
        )

        assert any(c[0] == "edit_file" for c in calls), "Should use edit_file"
        with open(filepath, encoding="utf-8") as f:
            content = f.read()
        assert "debug=true" in content

    print(f"Tool calls: {len(calls)}")
    print("PASS: test_v1_edit_file")
    return True


def test_v1_read_edit_verify():
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "version.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("version=1.0.0")

        response, calls, _ = run_agent_loop(
            client,
            f"1. Read {filepath}, 2. Change version to 2.0.0, 3. Read it again to verify.",
            V1_TOOLS,
            workdir=tmpdir,
        )

        tool_names = [c[0] for c in calls]
        assert "read_file" in tool_names, "Should read file"
        assert "edit_file" in tool_names or "write_file" in tool_names, "Should modify file"

        with open(filepath, encoding="utf-8") as f:
            content = f.read()
        assert "2.0.0" in content
        assert response is not None

    print(f"Tool calls: {len(calls)}")
    print("PASS: test_v1_read_edit_verify")
    return True


# =============================================================================
# v2 Tests: Todo Tracking
# =============================================================================

def test_v2_todo_single_task():
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        system = (
            f"You are a coding agent at {tmpdir}.\n"
            "Use TodoWrite to track tasks. Use write_file to create files. Be concise."
        )

        _, calls, _ = run_agent_loop(
            client,
            f"Create a file at {tmpdir}/hello.txt with content 'hello'. "
            "First use TodoWrite to plan, then use write_file to create the file.",
            V2_TOOLS,
            workdir=tmpdir,
            system_prompt=system,
            max_turns=10,
        )

        todo_calls = [c for c in calls if c[0] == "TodoWrite"]
        write_calls = [c for c in calls if c[0] == "write_file"]
        file_exists = os.path.exists(os.path.join(tmpdir, "hello.txt"))

        print(
            f"TodoWrite calls: {len(todo_calls)}, write_file calls: {len(write_calls)}"
        )
        assert file_exists or len(write_calls) >= 1, "Should attempt to create file"

    print(f"Tool calls: {len(calls)}")
    print("PASS: test_v2_todo_single_task")
    return True


def test_v2_todo_multi_step():
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        system = (
            f"You are a coding agent at {tmpdir}.\n"
            "Use TodoWrite to plan multi-step tasks. Use write_file to create files. "
            "Complete all steps."
        )

        _, calls, _ = run_agent_loop(
            client,
            f"""Create 3 files in {tmpdir}:
1. Use write_file to create a.txt with content 'A'
2. Use write_file to create b.txt with content 'B'
3. Use write_file to create c.txt with content 'C'
Use TodoWrite to track progress. Execute all steps.""",
            V2_TOOLS,
            workdir=tmpdir,
            system_prompt=system,
            max_turns=25,
        )

        files_created = sum(
            1
            for f in ["a.txt", "b.txt", "c.txt"]
            if os.path.exists(os.path.join(tmpdir, f))
        )
        write_calls = [c for c in calls if c[0] == "write_file"]
        todo_calls = [c for c in calls if c[0] == "TodoWrite"]

        print(
            f"Files created: {files_created}/3, write_file calls: {len(write_calls)}, "
            f"TodoWrite calls: {len(todo_calls)}"
        )
        assert files_created >= 2 or len(write_calls) >= 2

    print(f"Tool calls: {len(calls)}")
    print("PASS: test_v2_todo_multi_step")
    return True


# =============================================================================
# Error Handling Tests
# =============================================================================

def test_error_file_not_found():
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        response, calls, _ = run_agent_loop(
            client,
            f"Read the file {tmpdir}/nonexistent.txt and tell me if it exists.",
            V1_TOOLS,
            workdir=tmpdir,
        )

        assert response is not None
        assert any(word in response.lower() for word in ["not", "error", "exist", "found", "cannot"])

    print(f"Tool calls: {len(calls)}")
    print("PASS: test_error_file_not_found")
    return True


def test_error_command_fails():
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    response, calls, _ = run_agent_loop(
        client,
        "Run the command 'nonexistent_command_xyz' and tell me what happens.",
        [BASH_TOOL],
    )

    assert response is not None
    assert any(word in response.lower() for word in ["not found", "error", "fail", "command"])

    print(f"Tool calls: {len(calls)}")
    print("PASS: test_error_command_fails")
    return True


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    tests = [
        test_v0_bash_echo,
        test_v0_bash_pipeline,
        test_v1_read_file,
        test_v1_write_file,
        test_v1_edit_file,
        test_v1_read_edit_verify,
        test_v2_todo_single_task,
        test_v2_todo_multi_step,
        test_error_file_not_found,
        test_error_command_fails,
    ]

    failed = []
    for test_fn in tests:
        name = test_fn.__name__
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print("=" * 60)
        try:
            if not test_fn():
                failed.append(name)
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback

            traceback.print_exc()
            failed.append(name)

    print(f"\n{'='*60}")
    print(f"Results: {len(tests) - len(failed)}/{len(tests)} passed")
    print("=" * 60)

    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)
    else:
        print("All integration tests passed!")
        sys.exit(0)
